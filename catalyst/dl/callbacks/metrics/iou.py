from typing import Dict

import numpy as np

from catalyst.dl.core import (
    Callback, CallbackOrder, MetricCallback, RunnerState
)
from catalyst.dl.utils import criterion
from catalyst.utils.confusion_matrix import (
    calculate_confusion_matrix_from_tensors, calculate_tp_fp_fn
)


class IouCallback(MetricCallback):
    """
    IoU (Jaccard) metric callback.
    """

    def __init__(
        self,
        input_key: str = "targets",
        output_key: str = "logits",
        prefix: str = "iou",
        eps: float = 1e-7,
        threshold: float = None,
        activation: str = "Sigmoid",
    ):
        """
        Args:
            input_key (str): input key to use for iou calculation
                specifies our ``y_true``.
            output_key (str): output key to use for iou calculation;
                specifies our ``y_pred``
            prefix (str): key to store in logs
            eps (float): epsilon to avoid zero division
            threshold (float): threshold for outputs binarization
            activation (str): An torch.nn activation applied to the outputs.
                Must be one of ['none', 'Sigmoid', 'Softmax2d']
        """
        super().__init__(
            prefix=prefix,
            metric_fn=criterion.iou,
            input_key=input_key,
            output_key=output_key,
            eps=eps,
            threshold=threshold,
            activation=activation
        )


JaccardCallback = IouCallback


def calculate_jaccard(
    true_positives: np.array,
    false_positives: np.array,
    false_negatives: np.array
) -> np.array:
    """Calculate list of Jaccard indices.

    Args:
        true_positives:
        false_positives:
        false_negatives:

    Returns:

    """
    epsilon = 1e-7

    jaccard = (true_positives + epsilon) / (
        true_positives + false_positives + false_negatives + epsilon
    )

    if not np.all(jaccard <= 1):
        raise ValueError("Jaccard index should be less or equal to 1")

    if not np.all(jaccard > 0):
        raise ValueError("Jaccard index should be more than 1")

    return jaccard


class MulticlassIOUMetricCallback(Callback):
    def __init__(
        self,
        prefix: str = "jaccard",
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

    def on_batch_end(self, state: RunnerState):
        outputs = state.output[self.output_key]
        targets = state.input[self.input_key]

        confusion_matrix = calculate_confusion_matrix_from_tensors(
            outputs, targets
        )

        if self.confusion_matrix is None:
            self.confusion_matrix = confusion_matrix
        else:
            self.confusion_matrix += confusion_matrix

    def on_loader_end(self, state: RunnerState):

        tp_fp_fn_dict = calculate_tp_fp_fn(self.confusion_matrix)

        batch_metrics: Dict = calculate_jaccard(**tp_fp_fn_dict)

        for metric_id, jaccard_value in batch_metrics.items():
            if metric_id not in self.class_names:
                continue

            metric_name = self.class_names[metric_id]
            state.metrics.epoch_values[state.loader_name][
                f"{self.class_prefix}_{metric_name}"
            ] = jaccard_value

        state.metrics.epoch_values[state.loader_name]["mean"] = np.mean(
            [x for x in batch_metrics.values()]
        )

        self._reset_stats()


MulticlassJaccardMetricCallback = MulticlassIOUMetricCallback

__all__ = [
    "IouCallback", "JaccardCallback",
    "MulticlassIOUMetricCallback", "MulticlassJaccardMetricCallback"
]
