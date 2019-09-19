import logging
from typing import Optional, Any, List, Union

import torch
from catalyst.dl.core import Callback, RunnerState, CallbackOrder

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
        input_key: Union[str, List[str]] = "targets",
        output_key: Union[str, List[str]] = "logits",
        prefix: str = "loss",
        criterion_key: str = None,
        multiplier: float = 1.0
    ):
        """
        Args:
            input_key (Union[str, List[str]]): key or list of keys that takes
                values from the input dictionary
                If None, the whole input will be passed to the criterion.
            output_key (Union[str, List[str]]): key or list of keys that takes
                values from the output dictionary
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

    @staticmethod
    def _get(dictionary: dict, keys: Optional[Union[str, List[str]]]) -> Any:
        if keys is None:
            result = dictionary
        elif isinstance(keys, list):
            result = {key: dictionary[key] for key in keys}
        else:
            result = dictionary[keys]
        return result

    def _compute_loss(self, state: RunnerState, criterion):
        output = self._get(state.output, self.output_key)
        input = self._get(state.input, self.input_key)

        loss = criterion(output, input)
        return loss

    def on_stage_start(self, state: RunnerState):
        assert state.criterion is not None

    def on_batch_end(self, state: RunnerState):
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


class CriterionAggregatorCallback(Callback):
    """
    This callback allows you to aggregate the values of the loss
    (by ``sum`` or ``mean``) and put the value back into ``state.loss``.
    """
    def __init__(
        self,
        prefix: str,
        loss_keys: Union[str, List[str]] = None,
        loss_aggregate_fn: str = "sum",
        multiplier: float = 1.0
    ) -> None:
        """
        Args:
            prefix (str): new key for aggregated loss.
            loss_keys (List[str]): If not empty, it aggregates
                only the values from the loss by these keys.
            loss_aggregate_fn (str): function for aggregation.
                Must be either ``sum`` or ``mean``.
            multiplier (float): scale factor for the aggregated loss.
        """
        super().__init__(CallbackOrder.Criterion + 1)
        assert prefix is not None and isinstance(prefix, str), \
            "prefix must be str"
        self.prefix = prefix

        if isinstance(loss_keys, str):
            loss_keys = [loss_keys]
        self.loss_keys = loss_keys

        self.multiplier = multiplier
        if loss_aggregate_fn == "sum":
            self.loss_fn = lambda x: torch.sum(torch.stack(x)) * multiplier
        elif loss_aggregate_fn == "mean":
            self.loss_fn = lambda x: torch.mean(torch.stack(x)) * multiplier
        else:
            raise ValueError("loss_aggregate_fn must be `sum` or `mean`")

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
                result = [loss[key] for key in self.loss_keys]
            else:
                result = list(loss.values())
        else:
            result = [loss]

        return result

    def on_batch_end(self, state: RunnerState) -> None:
        loss = state.get_key(key="loss")
        loss = self._preprocess_loss(loss)
        loss = self.loss_fn(loss)

        state.metrics.add_batch_value(
            metrics_dict={
                self.prefix: loss.item(),
            }
        )

        _add_loss_to_state(self.prefix, state, loss)


__all__ = ["CriterionCallback", "CriterionAggregatorCallback"]
