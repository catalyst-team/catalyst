from typing import List

from catalyst.core import MetricCallback
from catalyst.utils import metrics


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
        activation: str = "none",
    ):
        """
        Args:
            input_key (str): input key to use for auc calculation
                specifies our ``y_true``
            output_key (str): output key to use for auc calculation;
                specifies our ``y_pred``
            prefix (str): name to display for mrr when printing
            activation (str): An torch.nn activation applied to the outputs.
                Must be one of ``'none'``, ``'Sigmoid'``, or ``'Softmax2d'``
        """
        super().__init__(
            prefix=prefix,
            metric_fn=metrics.mrr,
            input_key=input_key,
            output_key=output_key,
            activation=activation,
        )


__all__ = ["MRRCallback"]
