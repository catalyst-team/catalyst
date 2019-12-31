from typing import Any, Dict, List, Optional, Union  # isort:skip
import logging

import torch

from catalyst import utils
from catalyst.dl.core import Callback, CallbackOrder, RunnerState

logger = logging.getLogger(__name__)


def _add_loss_to_state(
    loss_key: Optional[str],
    state: RunnerState,
    loss: torch.Tensor
):
    if loss_key is None:
        if state.loss is not None:
            if isinstance(state.loss, list):
                state.loss.append(loss)
            else:
                state.loss = [state.loss, loss]
        else:
            state.loss = loss
    else:
        if state.loss is not None:
            assert isinstance(state.loss, dict)
            state.loss[loss_key] = loss
        else:
            state.loss = {loss_key: loss}


class CriterionCallback(Callback):
    """
    Callback for that measures loss with specified criterion.
    """
    def __init__(
        self,
        input_key: Union[str, List[str], Dict[str, str]] = "targets",
        output_key: Union[str, List[str], Dict[str, str]] = "logits",
        prefix: str = "loss",
        criterion_key: str = None,
        multiplier: float = 1.0
    ):
        """
        Args:
            input_key (Union[str, List[str], Dict[str, str]]): key/list/dict
                of keys that takes values from the input dictionary
                If None, the whole input will be passed to the criterion.
            output_key (Union[str, List[str], Dict[str, str]]): key/list/dict
                of keys that takes values from the input dictionary
                If None, the whole output will be passed to the criterion.
            prefix (str): prefix for metrics and output key for loss
                in ``state.loss`` dictionary
            criterion_key (str): A key to take a criterion in case
                there are several of them and they are in a dictionary format.
            multiplier (float): scale factor for the output loss.
        """
        super().__init__(CallbackOrder.Criterion)
        self.input_key = input_key
        self.output_key = output_key
        self.prefix = prefix
        self.criterion_key = criterion_key
        self.multiplier = multiplier

        self._get_input = utils.get_dictkey_auto_fn(self.input_key)
        self._get_output = utils.get_dictkey_auto_fn(self.output_key)
        kv_types = (dict, tuple, list, type(None))
        # @TODO: fix to only KV usage
        if isinstance(self.input_key, str) \
                and isinstance(self.output_key, str):
            self._compute_loss = self._compute_loss_value
        elif isinstance(self.input_key, kv_types) \
                and isinstance(self.output_key, kv_types):
            self._compute_loss = self._compute_loss_key_value
        else:
            raise NotImplementedError()

    def _compute_loss_value(self, state: RunnerState, criterion):
        output = self._get_output(state.output, self.output_key)
        input = self._get_input(state.input, self.input_key)

        loss = criterion(output, input)
        return loss

    def _compute_loss_key_value(self, state: RunnerState, criterion):
        output = self._get_output(state.output, self.output_key)
        input = self._get_input(state.input, self.input_key)

        loss = criterion(**output, **input)
        return loss

    def on_stage_start(self, state: RunnerState):
        """
        Checks that the current stage has correct criterion
        """
        assert state.criterion is not None

    def on_batch_end(self, state: RunnerState):
        """
        Computes the loss and add it to the metrics
        """
        criterion = state.get_key(
            key="criterion", inner_key=self.criterion_key
        )

        loss = self._compute_loss(state, criterion) * self.multiplier

        state.metrics.add_batch_value(
            metrics_dict={
                self.prefix: loss.item(),
            }
        )

        _add_loss_to_state(self.prefix, state, loss)


class CriterionOutputOnlyCallback(CriterionCallback):
    """
    Callback for that measures loss with specified criterion.
    Based on model output only.
    @TODO: merge logic with CriterionCallback.
    """
    def __init__(
        self,
        output_key: Union[Dict[str, str], List[str]],
        **kwargs
    ):
        """

        Args:
            output_key (Union[List[str]], Dict[str, str]): dict or list of keys
                that takes values from the output dictionary
                If None, the whole output will be passed to the criterion.
            **kwargs: CriterionCallback init parameters
        """
        super().__init__(
            input_key=None,
            output_key=output_key,
            **kwargs
        )

    def _compute_loss_value(self, state: RunnerState, criterion):
        output = self._get_output(state.output, self.output_key)

        loss = criterion(output)
        return loss

    def _compute_loss_key_value(self, state: RunnerState, criterion):
        output = self._get_output(state.output, self.output_key)

        loss = criterion(**output)
        return loss


class CriterionAggregatorCallback(Callback):
    """
    This callback allows you to aggregate the values of the loss
    (with different aggregation strategies)
    and put the value back into ``state.loss``.
    """
    def __init__(
        self,
        prefix: str,
        loss_keys: Union[str, List[str], Dict[str, float]] = None,
        loss_aggregate_fn: str = "sum",
        multiplier: float = 1.0
    ) -> None:
        """
        Args:
            prefix (str): new key for aggregated loss.
            loss_keys (Union[str, List[str], Dict[str, float]]): If not empty,
                it aggregates only the values from the loss by these keys.
                for ``weighted_sum`` aggregation it must be a Dict[str, float].
            loss_aggregate_fn (str): function for aggregation.
                Must be either ``sum``, ``mean`` or ``weighted_sum``.
            multiplier (float): scale factor for the aggregated loss.
        """
        super().__init__(CallbackOrder.Criterion + 1)
        if prefix is None or not isinstance(prefix, str):
            raise ValueError("prefix must be str")
        self.prefix = prefix

        if isinstance(loss_keys, str):
            loss_keys = [loss_keys]
        self.loss_keys = loss_keys

        self.multiplier = multiplier
        if loss_aggregate_fn == "sum":
            self.loss_fn = lambda x: torch.sum(torch.stack(x)) * multiplier
        elif loss_aggregate_fn == "weighted_sum":
            if loss_keys is None or not isinstance(loss_keys, dict):
                raise ValueError(
                    "For `weighted_sum` mode the loss_keys must be specified "
                    "and must be a dict"
                )
            self.loss_fn = lambda x: torch.sum(torch.stack(x)) * multiplier
        elif loss_aggregate_fn == "mean":
            self.loss_fn = lambda x: torch.mean(torch.stack(x)) * multiplier
        else:
            raise ValueError(
                "loss_aggregate_fn must be `sum`, `mean` or weighted_sum`"
            )

        self.loss_aggregate_name = loss_aggregate_fn

    def _preprocess_loss(self, loss: Any) -> List[torch.Tensor]:
        if isinstance(loss, list):
            if self.loss_keys is not None:
                logger.warning(
                    f"Trying to get {self.loss_keys} keys from the losses, "
                    "but the loss is a list. All values will be aggregated."
                )
            result = loss
        elif isinstance(loss, dict):
            if self.loss_keys is not None:
                if self.loss_aggregate_name == "weighted_sum":
                    result = [
                        loss[key] * value
                        for key, value in self.loss_keys.items()
                    ]
                else:
                    result = [loss[key] for key in self.loss_keys]
            else:
                result = list(loss.values())
        else:
            result = [loss]

        return result

    def on_batch_end(self, state: RunnerState) -> None:
        """
        Computes the loss and add it to the metrics
        """
        loss = state.get_key(key="loss")
        loss = self._preprocess_loss(loss)
        loss = self.loss_fn(loss)

        state.metrics.add_batch_value(
            metrics_dict={
                self.prefix: loss.item(),
            }
        )

        _add_loss_to_state(self.prefix, state, loss)


__all__ = [
    "CriterionCallback",
    "CriterionOutputOnlyCallback",
    "CriterionAggregatorCallback",
]
