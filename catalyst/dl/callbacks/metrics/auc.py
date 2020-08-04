from typing import List

from catalyst.core.callbacks import LoaderMetricCallback
from catalyst.utils import metrics
from catalyst.utils.metrics.functional import wrap_class_metric2dict


class AUCCallback(LoaderMetricCallback):
    """Calculates the AUC  per class for each loader.

    .. note::
        Currently, supports binary and multi-label cases.
    """

    def __init__(
        self,
        input_key: str = "targets",
        output_key: str = "logits",
        prefix: str = "auc",
        multiplier: float = 1.0,
        class_args: List[str] = None,
        **kwargs,
    ):
        """
        Args:
            input_key (str): input key to use for auc calculation
                specifies our ``y_true``.
            output_key (str): output key to use for auc calculation;
                specifies our ``y_pred``.
            prefix (str): metric's name.
            multiplier (float): scale factor for the metric.
            class_args (List[str]): class names to display in the logs.
                If None, defaults to indices for each class, starting from 0
        """
        super().__init__(
            prefix=prefix,
            metric_fn=wrap_class_metric2dict(
                metrics.auc, class_args=class_args
            ),
            input_key=input_key,
            output_key=output_key,
            multiplier=multiplier,
            **kwargs,
        )


__all__ = ["AUCCallback"]
