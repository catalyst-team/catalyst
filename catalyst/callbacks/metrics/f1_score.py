from typing import List

from catalyst.callbacks.metric import BatchMetricCallback
from catalyst.metrics.f1_score import fbeta_score
from catalyst.metrics.functional import (
    wrap_class_metric2dict,
    wrap_metric_fn_with_activation,
)


class F1ScoreCallback(BatchMetricCallback):
    """F1 score metric callback."""

    def __init__(
        self,
        input_key: str = "targets",
        output_key: str = "logits",
        prefix: str = "f1_score",
        activation: str = "Softmax",
        class_args: List[str] = None,
        **kwargs,
    ):
        """
        Args:
            input_key: input key to use for iou calculation
                specifies our ``y_true``
            output_key: output key to use for iou calculation;
                specifies our ``y_pred``
            prefix: key to store in logs
            activation: An torch.nn activation applied to the outputs.
                Must be one of ``'none'``, ``'Sigmoid'``, or ``'Softmax'``
        """
        metric_fn = wrap_metric_fn_with_activation(
            metric_fn=fbeta_score, activation=activation
        )
        metric_fn = wrap_class_metric2dict(metric_fn, class_args=class_args)
        super().__init__(
            prefix=prefix,
            metric_fn=metric_fn,
            input_key=input_key,
            output_key=output_key,
            **kwargs,
        )


__all__ = ["F1ScoreCallback"]
