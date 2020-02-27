from typing import Any, Dict, List, Optional, Union  # isort:skip
import logging

import torch

from catalyst import utils
from catalyst.core import _State, Callback, CallbackOrder

logger = logging.getLogger(__name__)


def _add_loss_to_state(
    loss_key: Optional[str], state: _State, loss: torch.Tensor
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
        if hasattr(self, "_compute_loss"):
            pass  # overridden in descendants
        elif isinstance(self.input_key, str) \
                and isinstance(self.output_key, str):
            self._compute_loss = self._compute_loss_value
        elif isinstance(self.input_key, kv_types) \
                and isinstance(self.output_key, kv_types):
            self._compute_loss = self._compute_loss_key_value
        else:
            raise NotImplementedError()

    def _compute_loss_value(self, state: _State, criterion):
        output = self._get_output(state.batch_out, self.output_key)
        input = self._get_input(state.batch_in, self.input_key)

        loss = criterion(output, input)
        return loss

    def _compute_loss_key_value(self, state: _State, criterion):
        output = self._get_output(state.batch_out, self.output_key)
        input = self._get_input(state.batch_in, self.input_key)

        loss = criterion(**output, **input)
        return loss

    def on_stage_start(self, state: _State):
        """
        Checks that the current stage has correct criterion
        """
        assert state.criterion is not None

    def on_batch_end(self, state: _State):
        """
        Computes the loss and add it to the metrics
        """
        criterion = state.get_attr(
            key="criterion", inner_key=self.criterion_key
        )

        loss = self._compute_loss(state, criterion) * self.multiplier

        state.metric_manager.add_batch_value(
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
    def __init__(self, output_key: Union[Dict[str, str], List[str]], **kwargs):
        """

        Args:
            output_key (Union[List[str]], Dict[str, str]): dict or list of keys
                that takes values from the output dictionary
                If None, the whole output will be passed to the criterion.
            **kwargs: CriterionCallback init parameters
        """
        super().__init__(input_key=None, output_key=output_key, **kwargs)

    def _compute_loss_value(self, state: _State, criterion):
        output = self._get_output(state.batch_out, self.output_key)

        loss = criterion(output)
        return loss

    def _compute_loss_key_value(self, state: _State, criterion):
        output = self._get_output(state.batch_out, self.output_key)

        loss = criterion(**output)
        return loss


__all__ = [
    "CriterionCallback", "CriterionOutputOnlyCallback",
]
