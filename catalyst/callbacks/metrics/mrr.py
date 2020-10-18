from catalyst.callbacks.metric import MetricCallback
from catalyst.metrics import mrr


class MRRCallback(MetricCallback):
    """Calculates the AUC  per class for each loader.

    .. note::
        Currently, supports binary and multi-label cases.
    """

    def __init__(
        self,
        input_key: str = "targets",
        output_key: str = "logits",
        prefix: str = "mrr",
    ):
        """
        Args:
            input_key (str): input key to use for mrr calculation
                specifies our ``y_true``
            output_key (str): output key to use for mrr calculation;
                specifies our ``y_pred``
            prefix (str): name to display for mrr when printing
        """
        super().__init__(
            prefix=prefix,
            metric_fn=mrr,
            input_key=input_key,
            output_key=output_key,
        )


__all__ = ["MRRCallback"]
