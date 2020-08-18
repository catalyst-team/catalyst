from typing import List

from catalyst.core import BatchMetricCallback
from catalyst.utils import metrics
from catalyst.utils.metrics.functional import (
    get_default_topk_args,
    wrap_topk_metric2dict,
)


class AccuracyCallback(BatchMetricCallback):
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
        multiplier: float = 1.0,
        topk_args: List[int] = None,
        num_classes: int = None,
        accuracy_args: List[int] = None,
        **kwargs,
    ):
        """
        Args:
            input_key (str): input key to use for accuracy calculation;
                specifies our `y_true`
            output_key (str): output key to use for accuracy calculation;
                specifies our `y_pred`
            prefix (str): key for the metric's name
            topk_args (List[int]): specifies which accuracy@K to log:
                [1] - accuracy
                [1, 3] - accuracy at 1 and 3
                [1, 3, 5] - accuracy at 1, 3 and 5
            num_classes (int): number of classes to calculate ``topk_args``
                if ``accuracy_args`` is None
            activation (str): An torch.nn activation applied to the outputs.
                Must be one of ``"none"``, ``"Sigmoid"``, or ``"Softmax"``
        """
        topk_args = (
            topk_args or accuracy_args or get_default_topk_args(num_classes)
        )

        super().__init__(
            prefix=prefix,
            metric_fn=wrap_topk_metric2dict(
                metrics.accuracy, topk_args=topk_args
            ),
            input_key=input_key,
            output_key=output_key,
            multiplier=multiplier,
            **kwargs,
        )


class MultiLabelAccuracyCallback(BatchMetricCallback):
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
