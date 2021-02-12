from catalyst.callbacks.metric import LoaderMetricCallback
from catalyst.metrics.auc import AUCMetric


class AUCCallback(LoaderMetricCallback):
    """ROC-AUC  metric callback."""

    def __init__(
        self, input_key: str, target_key: str, prefix: str = None, suffix: str = None,
    ):
        """
        Args:
            input_key: input key to use for auc calculation
                specifies our ``y_true``.
            target_key: output key to use for auc calculation;
                specifies our ``y_pred``.
        """
        super().__init__(
            metric=AUCMetric(prefix=prefix, suffix=suffix),
            input_key=input_key,
            target_key=target_key,
        )


__all__ = ["AUCCallback"]
