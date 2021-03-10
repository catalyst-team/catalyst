from typing import List, Optional

from catalyst.callbacks.metric import BatchMetricCallback
from catalyst.metrics._segmentation import DiceMetric, IOUMetric, TrevskyMetric


class IOUCallback(BatchMetricCallback):
    """IOU metric callback."""

    def __init__(
        self,
        input_key: str,
        target_key: str,
        prefix: str = "iou",
        suffix: str = None,
        class_dim: int = 1,
        weights: Optional[List[float]] = None,
        class_names: Optional[List[str]] = None,
        threshold: Optional[float] = None,
    ):
        """
        Args:
            input_key: input key to use for metric calculation, specifies our `y_pred`
            target_key: output key to use for metric calculation, specifies our `y_true`
            prefix: key for the metric's name
        """
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
        )


JaccardCallback = IOUCallback


class DiceCallback(BatchMetricCallback):
    """Dice metric callback."""

    def __init__(
        self,
        input_key: str,
        target_key: str,
        prefix: str = "dice",
        suffix: str = None,
        class_dim: int = 1,
        weights: Optional[List[float]] = None,
        class_names: Optional[List[str]] = None,
        threshold: Optional[float] = None,
    ):
        """
        Args:
            input_key: input key to use for metric calculation, specifies our `y_pred`
            target_key: output key to use for metric calculation, specifies our `y_true`
            prefix: key for the metric's name
        """
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
        )


class TrevskyCallback(BatchMetricCallback):
    """Trevsky metric callback."""

    def __init__(
        self,
        input_key: str,
        target_key: str,
        alpha: float,
        beta: Optional[float] = None,
        prefix: str = "trevsky",
        suffix: str = None,
        class_dim: int = 1,
        weights: Optional[List[float]] = None,
        class_names: Optional[List[str]] = None,
        threshold: Optional[float] = None,
    ):
        """
        Args:
            input_key: input key to use for metric calculation, specifies our `y_pred`
            target_key: output key to use for metric calculation, specifies our `y_true`
            prefix: key for the metric's name
        """
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
        )


__all__ = ["IOUCallback", "JaccardCallback", "DiceCallback", "TrevskyCallback"]
