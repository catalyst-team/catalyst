from typing import List

from catalyst.core import MetricCallback, MultiMetricCallback
from catalyst.dl.callbacks.metrics.functional import get_default_topk_args
from catalyst.utils import metrics


class AccuracyCallback(MultiMetricCallback):
    """Accuracy metric callback.
    Computes multi-class accuracy@topk for the specified values of `topk`.

    .. note::
        For multi-label accuracy please use
        `catalyst.dl.callbacks.metrics.MultiLabelAccuracyCallback`
    """

    def __init__(
        self,
        input_key: str = "targets",
        output_key: str = "logits",
        prefix: str = "accuracy",
        accuracy_args: List[int] = None,
        num_classes: int = None,
        activation: str = None,
    ):
        """
        Args:
            input_key (str): input key to use for accuracy calculation;
                specifies our `y_true`
            output_key (str): output key to use for accuracy calculation;
                specifies our `y_pred`
            prefix (str): key for the metric's name
            accuracy_args (List[int]): specifies which accuracy@K to log:
                [1] - accuracy
                [1, 3] - accuracy at 1 and 3
                [1, 3, 5] - accuracy at 1, 3 and 5
            num_classes (int): number of classes to calculate ``accuracy_args``
                if ``accuracy_args`` is None
            activation (str): An torch.nn activation applied to the outputs.
                Must be one of ``"none"``, ``"Sigmoid"``, or ``"Softmax"``
        """
        list_args = accuracy_args or get_default_topk_args(num_classes)

        super().__init__(
            prefix=prefix,
            metric_fn=metrics.accuracy,
            list_args=list_args,
            input_key=input_key,
            output_key=output_key,
            topk=list_args,
            activation=activation,
        )


class MultiLabelAccuracyCallback(MetricCallback):
    """Accuracy metric callback.
    Computes multi-class accuracy@topk for the specified values of `topk`.

    .. note::
        For multi-label accuracy please use
        `catalyst.dl.callbacks.metrics.MultiLabelAccuracyCallback`
    """

    def __init__(
        self,
        input_key: str = "targets",
        output_key: str = "logits",
        prefix: str = "multi_label_accuracy",
        threshold: float = None,
        activation: str = "Sigmoid",
    ):
        """
        Args:
            input_key (str): input key to use for accuracy calculation;
                specifies our `y_true`
            output_key (str): output key to use for accuracy calculation;
                specifies our `y_pred`
            prefix (str): key for the metric's name
            threshold (float): threshold for for model output
            activation (str): An torch.nn activation applied to the outputs.
                Must be one of ``"none"``, ``"Sigmoid"``, or ``"Softmax"``
        """
        super().__init__(
            prefix=prefix,
            metric_fn=metrics.multi_label_accuracy,
            input_key=input_key,
            output_key=output_key,
            threshold=threshold,
            activation=activation,
        )


__all__ = ["AccuracyCallback", "MultiLabelAccuracyCallback"]
