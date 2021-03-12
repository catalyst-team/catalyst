from typing import List, Optional

from catalyst.callbacks.metric import BatchMetricCallback
from catalyst.metrics._segmentation import DiceMetric, IOUMetric, TrevskyMetric


class IOUCallback(BatchMetricCallback):
    """IOU metric callback.

    Args:
        input_key: input key to use for metric calculation, specifies our `y_pred`
        target_key: output key to use for metric calculation, specifies our `y_true`
        class_dim: @TODO: docs
        weights: @TODO: docs
        class_names: @TODO: docs
        threshold: @TODO: docs
        log_on_batch: boolean flag to log computed metrics every batch
        prefix: metric prefix
        suffix: metric suffix
    """

    def __init__(
        self,
        input_key: str,
        target_key: str,
        class_dim: int = 1,
        weights: Optional[List[float]] = None,
        class_names: Optional[List[str]] = None,
        threshold: Optional[float] = None,
        log_on_batch: bool = True,
        prefix: str = None,
        suffix: str = None,
    ):
        """Init."""
        super().__init__(
            metric=IOUMetric(
                class_dim=class_dim,
                weights=weights,
                class_names=class_names,
                threshold=threshold,
                prefix=prefix,
                suffix=suffix,
            ),
            input_key=input_key,
            target_key=target_key,
            log_on_batch=log_on_batch,
        )


class DiceCallback(BatchMetricCallback):
    """Dice metric callback.

    Args:
        input_key: input key to use for metric calculation, specifies our `y_pred`
        target_key: output key to use for metric calculation, specifies our `y_true`
        class_dim: @TODO: docs
        weights: @TODO: docs
        class_names: @TODO: docs
        threshold: @TODO: docs
        log_on_batch: boolean flag to log computed metrics every batch
        prefix: metric prefix
        suffix: metric suffix
    """

    def __init__(
        self,
        input_key: str,
        target_key: str,
        class_dim: int = 1,
        weights: Optional[List[float]] = None,
        class_names: Optional[List[str]] = None,
        threshold: Optional[float] = None,
        log_on_batch: bool = True,
        prefix: str = None,
        suffix: str = None,
    ):
        """Init."""
        super().__init__(
            metric=DiceMetric(
                class_dim=class_dim,
                weights=weights,
                class_names=class_names,
                threshold=threshold,
                prefix=prefix,
                suffix=suffix,
            ),
            input_key=input_key,
            target_key=target_key,
            log_on_batch=log_on_batch,
        )


class TrevskyCallback(BatchMetricCallback):
    """Trevsky metric callback.

    Args:
        input_key: input key to use for metric calculation, specifies our `y_pred`
        target_key: output key to use for metric calculation, specifies our `y_true`
        class_dim: @TODO: docs
        weights: @TODO: docs
        class_names: @TODO: docs
        threshold: @TODO: docs
        log_on_batch: boolean flag to log computed metrics every batch
        prefix: metric prefix
        suffix: metric suffix
    """

    def __init__(
        self,
        input_key: str,
        target_key: str,
        alpha: float,
        beta: Optional[float] = None,
        class_dim: int = 1,
        weights: Optional[List[float]] = None,
        class_names: Optional[List[str]] = None,
        threshold: Optional[float] = None,
        log_on_batch: bool = True,
        prefix: str = None,
        suffix: str = None,
    ):
        """Init."""
        super().__init__(
            metric=TrevskyMetric(
                alpha=alpha,
                beta=beta,
                class_dim=class_dim,
                weights=weights,
                class_names=class_names,
                threshold=threshold,
                prefix=prefix,
                suffix=suffix,
            ),
            input_key=input_key,
            target_key=target_key,
            log_on_batch=log_on_batch,
        )


__all__ = ["IOUCallback", "DiceCallback", "TrevskyCallback"]
