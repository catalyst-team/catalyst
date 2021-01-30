from catalyst.callbacks.metric import BatchMetricCallback
from catalyst.metrics import mrr


class MRRCallback(BatchMetricCallback):
    """Calculates the MRR."""

    def __init__(
        self,
        input_key: str = "targets",
        output_key: str = "logits",
        prefix: str = "mrr",
        multiplier: float = 1.0,
        **kwargs,
    ):
        """
        Args:
            input_key (str): input key to use for mrr calculation
                specifies our ``y_true``
            output_key (str): output key to use for mrr calculation;
                specifies our ``y_pred``
            prefix (str): name to display for mrr when printing
            **kwargs: key-value params to pass to the metric

        .. note::
            For `**kwargs` info, please follow
            `catalyst.metrics.mrr.mrr` docs

        """
        super().__init__(
            prefix=prefix,
            metric_fn=mrr,
            input_key=input_key,
            output_key=output_key,
            multiplier=multiplier,
            **kwargs,
        )


__all__ = ["MRRCallback"]
