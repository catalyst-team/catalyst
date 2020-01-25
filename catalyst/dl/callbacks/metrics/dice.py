from typing import Dict

import numpy as np

from catalyst.dl.core import Callback, CallbackOrder, MetricCallback, State
from catalyst.utils import criterion
from catalyst.utils.confusion_matrix import (
    calculate_confusion_matrix_from_tensors, calculate_tp_fp_fn
)
from .functional import calculate_dice


class DiceCallback(MetricCallback):
    """
    Dice metric callback.
    """

    def __init__(
        self,
        input_key: str = "targets",
        output_key: str = "logits",
        prefix: str = "dice",
        eps: float = 1e-7,
        threshold: float = None,
        activation: str = "Sigmoid"
    ):
        """
        Args:
            input_key (str): input key to use for dice calculation;
                specifies our `y_true`.
            output_key (str): output key to use for dice calculation;
                specifies our `y_pred`.
        """
        super().__init__(
            prefix=prefix,
            metric_fn=criterion.dice,
            input_key=input_key,
            output_key=output_key,
            eps=eps,
            threshold=threshold,
            activation=activation
        )


class MulticlassDiceMetricCallback(Callback):
    def __init__(
        self,
        prefix: str = "dice",
        input_key: str = "targets",
        output_key: str = "logits",
        class_names=None,
        class_prefix="",
        **metric_params
    ):
        super().__init__(CallbackOrder.Metric)
        self.prefix = prefix
        self.input_key = input_key
        self.output_key = output_key
        self.metric_params = metric_params
        self.confusion_matrix = None
        self.class_names = class_names  # dictionary {class_id: class_name}
        self.class_prefix = class_prefix

    def _reset_stats(self):
        self.confusion_matrix = None

    def on_batch_end(self, state: State):
        outputs = state.output[self.output_key]
        targets = state.input[self.input_key]

        confusion_matrix = calculate_confusion_matrix_from_tensors(
            outputs, targets
        )

        if self.confusion_matrix is None:
            self.confusion_matrix = confusion_matrix
        else:
            self.confusion_matrix += confusion_matrix

    def on_loader_end(self, state: State):
        tp_fp_fn_dict = calculate_tp_fp_fn(self.confusion_matrix)

        batch_metrics: Dict = calculate_dice(**tp_fp_fn_dict)

        for metric_id, dice_value in batch_metrics.items():
            if metric_id not in self.class_names:
                continue

            metric_name = self.class_names[metric_id]
            state.metric_manager.epoch_values[state.loader_name][
                f"{self.class_prefix}_{metric_name}"
            ] = dice_value

        state.metric_manager.epoch_values[state.loader_name]["mean"] = np.mean(
            [x for x in batch_metrics.values()]
        )

        self._reset_stats()


__all__ = ["DiceCallback", "MulticlassDiceMetricCallback"]
