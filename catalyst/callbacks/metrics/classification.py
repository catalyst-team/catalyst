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
        zero_division: value to set in case of zero division during metrics
            (precision, recall) computation; should be one of 0 or 1
        log_on_batch: boolean flag to log computed metrics every batch
        prefix: metric prefix
        suffix: metric suffix
    """

    def __init__(
        self,
        input_key: str,
        target_key: str,
        num_classes: int,
        zero_division: int = 0,
        log_on_batch: bool = True,
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
            log_on_batch=log_on_batch,
        )


class MultilabelPrecisionRecallF1SupportCallback(BatchMetricCallback):
    """Multilabel PrecisionRecallF1Support metric callback.

    Args:
        input_key: input key to use for metric calculation, specifies our `y_pred`
        target_key: output key to use for metric calculation, specifies our `y_true`
        num_classes: number of classes
        zero_division: value to set in case of zero division during metrics
            (precision, recall) computation; should be one of 0 or 1
        log_on_batch: boolean flag to log computed metrics every batch
        prefix: metric prefix
        suffix: metric suffix
    """

    def __init__(
        self,
        input_key: str,
        target_key: str,
        num_classes: int,
        zero_division: int = 0,
        log_on_batch: bool = True,
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
            log_on_batch=log_on_batch,
        )


__all__ = [
    "PrecisionRecallF1SupportCallback",
    "MultilabelPrecisionRecallF1SupportCallback",
]
