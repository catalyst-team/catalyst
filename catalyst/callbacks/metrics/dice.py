from typing import List, TYPE_CHECKING

import numpy as np

from catalyst.callbacks.metric import BatchMetricCallback
from catalyst.contrib.utils.torch_extra import (
    calculate_confusion_matrix_from_tensors,
    calculate_tp_fp_fn,
)
from catalyst.core.callback import Callback, CallbackOrder
from catalyst.metrics.dice import calculate_dice, dice
from catalyst.metrics.functional import wrap_class_metric2dict, wrap_metric_fn_with_activation

if TYPE_CHECKING:
    from catalyst.core.runner import IRunner


class DiceCallback(BatchMetricCallback):
    """Dice metric callback."""

    def __init__(
        self,
        input_key: str = "targets",
        output_key: str = "logits",
        prefix: str = "dice",
        activation: str = "Sigmoid",
        per_class: bool = False,
        class_args: List[str] = None,
        **kwargs,
    ):
        """
        Args:
            input_key: input key to use for iou calculation
                specifies our ``y_true``
            output_key: output key to use for iou calculation;
                specifies our ``y_pred``
            prefix: key to store in logs
            activation: An torch.nn activation applied to the outputs.
                Must be one of ``'none'``, ``'Sigmoid'``, or ``'Softmax'``
            per_class: boolean flag to log per class metrics,
                or use mean/macro statistics otherwise
            class_args: class names to display in the logs.
                If None, defaults to indices for each class, starting from 0
            **kwargs: key-value params to pass to the metric

        .. note::
            For ``**kwargs`` info, please follow
            ``catalyst.callbacks.metric.BatchMetricCallback`` and
            ``catalyst.metrics.dice.dice`` docs
        """
        metric_fn = wrap_metric_fn_with_activation(metric_fn=dice, activation=activation)
        metric_fn = wrap_class_metric2dict(metric_fn, per_class=per_class, class_args=class_args)
        super().__init__(
            prefix=prefix,
            metric_fn=metric_fn,
            input_key=input_key,
            output_key=output_key,
            **kwargs,
        )


class MulticlassDiceMetricCallback(Callback):
    """
    Global multiclass Dice Metric Callback: calculates the exact
    dice score across multiple batches. This callback is good for getting
    the dice score with small batch sizes where the batchwise dice is noisier.
    """

    def __init__(
        self,
        input_key: str = "targets",
        output_key: str = "logits",
        prefix: str = "dice",
        class_names=None,
    ):
        """
        Args:
            input_key: input key to use for dice calculation;
                specifies our `y_true`
            output_key: output key to use for dice calculation;
                specifies our `y_pred`
            prefix: prefix for printing the metric
            class_names: if dictionary, should be:
                {class_id: class_name, ...} where class_id is an integer
                This allows you to ignore class indices.
                if list, make sure it corresponds to the number of classes
        """
        super().__init__(CallbackOrder.metric)
        self.input_key = input_key
        self.output_key = output_key
        self.prefix = prefix
        self.confusion_matrix = None
        self.class_names = class_names

    def _reset_stats(self):
        """Resets the confusion matrix holding the epoch-wise stats."""
        self.confusion_matrix = None

    def on_batch_end(self, runner: "IRunner"):
        """Records the confusion matrix at the end of each batch.

        Args:
            runner: current runner
        """
        outputs = runner.output[self.output_key]
        targets = runner.input[self.input_key]

        confusion_matrix = calculate_confusion_matrix_from_tensors(outputs, targets)

        if self.confusion_matrix is None:
            self.confusion_matrix = confusion_matrix
        else:
            self.confusion_matrix += confusion_matrix

    def on_loader_end(self, runner: "IRunner"):
        """Logs dice scores to the ``loader_metrics``.

        Args:
            runner: current runner
        """
        tp_fp_fn_dict = calculate_tp_fp_fn(self.confusion_matrix)

        dice_scores: np.ndarray = calculate_dice(**tp_fp_fn_dict)

        # logging the dice scores in the state
        for i, dice_score in enumerate(dice_scores):
            if isinstance(self.class_names, dict) and i not in list(self.class_names.keys()):
                continue
            postfix = self.class_names[i] if self.class_names is not None else str(i)

            runner.loader_metrics[f"{self.prefix}_{postfix}"] = dice_score

        # For supporting averaging of only classes specified in `class_names`
        values_to_avg = [
            value
            for key, value in runner.loader_metrics.items()
            if key.startswith(f"{self.prefix}_")
        ]
        runner.loader_metrics[f"{self.prefix}_mean"] = np.mean(values_to_avg)

        self._reset_stats()


__all__ = [
    "DiceCallback",
    "MulticlassDiceMetricCallback",
]
