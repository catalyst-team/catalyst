from catalyst.callbacks.metric import BatchMetricCallback
from catalyst.metrics.f1_score import f1_score


class F1ScoreCallback(BatchMetricCallback):
    """F1 score metric callback."""

    def __init__(
        self,
        input_key: str = "targets",
        output_key: str = "logits",
        prefix: str = "f1_score",
        beta: float = 1.0,
        eps: float = 1e-7,
        threshold: float = None,
        activation: str = "Sigmoid",
    ):
        """
        Args:
            input_key: input key to use for iou calculation
                specifies our ``y_true``
            output_key: output key to use for iou calculation;
                specifies our ``y_pred``
            prefix: key to store in logs
            beta: beta param for f_score
            eps: epsilon to avoid zero division
            threshold: threshold for outputs binarization
            activation: An torch.nn activation applied to the outputs.
                Must be one of ``'none'``, ``'Sigmoid'``, or ``'Softmax2d'``
        """
        super().__init__(
            prefix=prefix,
            metric_fn=f1_score,
            input_key=input_key,
            output_key=output_key,
            beta=beta,
            eps=eps,
            threshold=threshold,
            activation=activation,
        )


__all__ = ["F1ScoreCallback"]
