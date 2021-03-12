from catalyst.callbacks.metric import BatchMetricCallback
from catalyst.metrics._classification import (
    MulticlassPrecisionRecallF1SupportMetric,
    MultilabelPrecisionRecallF1SupportMetric,
)


class PrecisionRecallF1SupportCallback(BatchMetricCallback):
    """Multiclass PrecisionRecallF1Support metric callback.

    Args:
        input_key: input key to use for metric calculation, specifies our `y_pred`
        target_key: output key to use for metric calculation, specifies our `y_true`
        num_classes: number of classes
        zero_division: @TODO: docs
        prefix: metric's prefix
        suffix: metric's suffix
    """

    def __init__(
        self,
        input_key: str,
        target_key: str,
        num_classes: int,
        zero_division: int = 0,
        prefix: str = None,
        suffix: str = None,
    ):
        """Init."""
        super().__init__(
            metric=MulticlassPrecisionRecallF1SupportMetric(
                num_classes=num_classes, zero_division=zero_division, prefix=prefix, suffix=suffix
            ),
            input_key=input_key,
            target_key=target_key,
        )


class MultilabelPrecisionRecallF1SupportCallback(BatchMetricCallback):
    """Multilabel PrecisionRecallF1Support metric callback.

    Args:
        input_key: input key to use for metric calculation, specifies our `y_pred`
        target_key: output key to use for metric calculation, specifies our `y_true`
        num_classes: number of classes
        zero_division: @TODO: docs
        prefix: metric's prefix
        suffix: metric's suffix
    """

    def __init__(
        self,
        input_key: str,
        target_key: str,
        num_classes: int,
        zero_division: int = 0,
        prefix: str = None,
        suffix: str = None,
    ):
        """Init."""
        super().__init__(
            metric=MultilabelPrecisionRecallF1SupportMetric(
                num_classes=num_classes, zero_division=zero_division, prefix=prefix, suffix=suffix
            ),
            input_key=input_key,
            target_key=target_key,
        )


__all__ = [
    "PrecisionRecallF1SupportCallback",
    "MultilabelPrecisionRecallF1SupportCallback",
]
