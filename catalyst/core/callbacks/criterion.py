from typing import Dict, List, Union  # isort:skip
import logging

from catalyst import utils
from catalyst.core import _State, Callback, CallbackOrder

logger = logging.getLogger(__name__)


class CriterionCallback(Callback):
    """
    Callback for that measures loss with specified criterion.
    """
    def __init__(
        self,
        input_key: Union[str, List[str], Dict[str, str]] = "targets",
        output_key: Union[str, List[str], Dict[str, str]] = "logits",
        loss_key: str = "loss",
        criterion_key: str = None,
        multiplier: float = 1.0
    ):
        """
        Args:
            input_key (Union[str, List[str], Dict[str, str]]): key/list/dict
                of keys that takes values from the input dictionary
                If '__all__', the whole input will be passed to the criterion
                If None, empty dict will be passed to the criterion.
            output_key (Union[str, List[str], Dict[str, str]]): key/list/dict
                of keys that takes values from the input dictionary
                If '__all__', the whole output will be passed to the criterion
                If None, empty dict will be passed to the criterion.
            loss_key (str): prefix for metrics and output key for loss
                in ``state.batch_metrics`` dictionary
            criterion_key (str): A key to take a criterion in case
                there are several of them and they are in a dictionary format.
            multiplier (float): scale factor for the output loss.
        """
        super().__init__(CallbackOrder.Metric)
        self.input_key = input_key
        self.output_key = output_key
        self.loss_key = loss_key
        self.criterion_key = criterion_key
        self.multiplier = multiplier

        self._get_input = utils.get_dictkey_auto_fn(self.input_key)
        self._get_output = utils.get_dictkey_auto_fn(self.output_key)
        kv_types = (dict, tuple, list, type(None))

        is_value_input = \
            isinstance(self.input_key, str) and self.input_key != "__all__"
        is_value_output = \
            isinstance(self.output_key, str) and self.output_key != "__all__"
        is_kv_input = \
            isinstance(self.input_key, kv_types) or self.input_key == "__all__"
        is_kv_output = (
            isinstance(self.output_key, kv_types)
            or self.output_key == "__all__"
        )

        # @TODO: fix to only KV usage
        if hasattr(self, "_compute_loss"):
            pass  # overridden in descendants
        elif is_value_input and is_value_output:
            self._compute_loss = self._compute_loss_value
        elif is_kv_input and is_kv_output:
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
        criterion = state.get_attr(
            key="criterion", inner_key=self.criterion_key
        )
        assert criterion is not None
        self._criterion = criterion

    def on_batch_end(self, state: _State):
        """
        Computes the loss and add it to the metrics
        """

        loss = self._compute_loss(state, self._criterion) * self.multiplier
        state.batch_metrics[self.loss_key] = loss


__all__ = [
    "CriterionCallback",
]
