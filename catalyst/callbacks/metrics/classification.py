from catalyst.callbacks.metric import BatchMetricCallback
from catalyst.metrics._classification import (
    MulticlassPrecisionRecallF1SupportMetric,
    MultilabelPrecisionRecallF1SupportMetric,
)


class MulticlassPrecisionRecallF1SupportCallback(BatchMetricCallback):
    """Multiclass PrecisionRecallF1Support metric callback."""

    def __init__(
        self,
        input_key: str,
        target_key: str,
        num_classes: int,
        zero_division: int = 0,
        prefix: str = None,
        suffix: str = None,
    ):
        """
        Args:
            input_key: input key to use for metric calculation, specifies our `y_pred`
            target_key: output key to use for metric calculation, specifies our `y_true`
            prefix: key for the metric's name
            num_classes: number of classes
        """
        super().__init__(
            metric=MulticlassPrecisionRecallF1SupportMetric(
                num_classes=num_classes, zero_division=zero_division, prefix=prefix, suffix=suffix
            ),
            input_key=input_key,
            target_key=target_key,
        )


class MultilabelPrecisionRecallF1SupportCallback(BatchMetricCallback):
    """Multilabel PrecisionRecallF1Support metric callback."""

    def __init__(
        self,
        input_key: str,
        target_key: str,
        num_classes: int,
        zero_division: int = 0,
        prefix: str = None,
        suffix: str = None,
    ):
        """
        Args:
            input_key: input key to use for metric calculation, specifies our `y_pred`
            target_key: output key to use for metric calculation, specifies our `y_true`
            prefix: key for the metric's name
            num_classes: number of classes
        """
        super().__init__(
            metric=MultilabelPrecisionRecallF1SupportMetric(
                num_classes=num_classes, zero_division=zero_division, prefix=prefix, suffix=suffix
            ),
            input_key=input_key,
            target_key=target_key,
        )


__all__ = [
    "MulticlassPrecisionRecallF1SupportCallback",
    "MultilabelPrecisionRecallF1SupportCallback",
]
