from typing import List

from catalyst.core.callbacks import LoaderMetricCallback
from catalyst.utils import metrics
from catalyst.utils.metrics.functional import wrap_class_metric2dict


class AveragePrecisionCallback(LoaderMetricCallback):
    """AveragePrecision metric callback."""

    def __init__(
        self,
        input_key: str = "targets",
        output_key: str = "logits",
        prefix: str = "average_precision",
        multiplier: float = 1.0,
        class_args: List[str] = None,
        **kwargs,
    ):
        """
        Args:
            input_key (str): input key to use for
                calculation mean average precision;
                specifies our `y_true`.
            output_key (str): output key to use for
                calculation mean average precision;
                specifies our `y_pred`.
            prefix (str): metric's name.
            multiplier (float): scale factor for the metric.
            class_args (List[str]): class names to display in the logs.
                If None, defaults to indices for each class, starting from 0
        """
        super().__init__(
            prefix=prefix,
            metric_fn=wrap_class_metric2dict(
                metrics.average_precision, class_args=class_args
            ),
            input_key=input_key,
            output_key=output_key,
            multiplier=multiplier,
            **kwargs,
        )


__all__ = ["AveragePrecisionCallback"]
