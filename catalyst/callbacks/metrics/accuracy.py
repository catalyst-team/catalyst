from typing import List

from catalyst.callbacks.metric import BatchMetricCallback
from catalyst.metrics.accuracy import accuracy, multilabel_accuracy
from catalyst.metrics.functional import (
    get_default_topk_args,
    wrap_metric_fn_with_activation,
    wrap_topk_metric2dict,
)


class AccuracyCallback(BatchMetricCallback):
    """Accuracy metric callback.
    Computes multiclass accuracy@topk for the specified values of `topk`.

    .. note::
        For multilabel accuracy please use
        `catalyst.callbacks.metrics.MultiLabelAccuracyCallback`
    """

    def __init__(
        self,
        input_key: str = "targets",
        output_key: str = "logits",
        prefix: str = "accuracy",
        topk_args: List[int] = None,
        num_classes: int = None,
        accuracy_args: List[int] = None,
        **kwargs,
    ):
        """
        Args:
            input_key: input key to use for accuracy calculation;
                specifies our `y_true`
            output_key: output key to use for accuracy calculation;
                specifies our `y_pred`
            prefix: key for the metric's name
            topk_args: specifies which accuracy@K to log:
                [1] - accuracy
                [1, 3] - accuracy at 1 and 3
                [1, 3, 5] - accuracy at 1, 3 and 5
            num_classes: number of classes to calculate ``topk_args``
                if ``accuracy_args`` is None
            **kwargs: key-value params to pass to the metric

        .. note::
            For ``**kwargs`` info, please follow
            ``catalyst.callbacks.metric.BatchMetricCallback`` and
            ``catalyst.metrics.accuracy.accuracy`` docs
        """
        topk_args = topk_args or accuracy_args or get_default_topk_args(num_classes)

        super().__init__(
            prefix=prefix,
            metric_fn=wrap_topk_metric2dict(accuracy, topk_args=topk_args),
            input_key=input_key,
            output_key=output_key,
            **kwargs,
        )


class MultiLabelAccuracyCallback(BatchMetricCallback):
    """Accuracy metric callback.
    Computes multiclass accuracy@topk for the specified values of `topk`.

    .. note::
        For multilabel accuracy please use
        `catalyst.callbacks.metrics.MultiLabelAccuracyCallback`
    """

    def __init__(
        self,
        input_key: str = "targets",
        output_key: str = "logits",
        prefix: str = "multilabel_accuracy",
        activation: str = "Sigmoid",
        threshold: float = None,
        **kwargs,
    ):
        """
        Args:
            input_key: input key to use for accuracy calculation;
                specifies our `y_true`
            output_key: output key to use for accuracy calculation;
                specifies our `y_pred`
            prefix: key for the metric's name
            activation: An torch.nn activation applied to the outputs.
                Must be one of ``"none"``, ``"Sigmoid"``, or ``"Softmax"``
            threshold: threshold for for model output
            **kwargs: key-value params to pass to the metric

        .. note::
            For ``**kwargs`` info, please follow
            ``catalyst.callbacks.metric.BatchMetricCallback`` and
            ``catalyst.metrics.accuracy.multilabel_accuracy`` docs
        """
        super().__init__(
            prefix=prefix,
            metric_fn=wrap_metric_fn_with_activation(
                metric_fn=multilabel_accuracy, activation=activation
            ),
            input_key=input_key,
            output_key=output_key,
            threshold=threshold,
            **kwargs,
        )


__all__ = ["AccuracyCallback", "MultiLabelAccuracyCallback"]
