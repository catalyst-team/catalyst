from catalyst.dl.core import MetricCallback
from catalyst.dl.utils import criterion


class F1ScoreCallback(MetricCallback):
    """
    F1 score metric callback.
    """

    def __init__(
        self,
        input_key: str = "targets",
        output_key: str = "logits",
        prefix: str = "f1_score",
        beta: float = 1.0,
        eps: float = 1e-7,
        threshold: float = None,
        activation: str = "Sigmoid"
    ):
        """
        Args:
            input_key (str): input key to use for iou calculation
                specifies our ``y_true``.
            output_key (str): output key to use for iou calculation;
                specifies our ``y_pred``
            prefix (str): key to store in logs
            beta (float): beta param for f_score
            eps (float): epsilon to avoid zero division
            threshold (float): threshold for outputs binarization
            activation (str): An torch.nn activation applied to the outputs.
                Must be one of ['none', 'Sigmoid', 'Softmax2d']
        """

        super().__init__(
            prefix=prefix,
            metric_fn=criterion.f1_score,
            input_key=input_key,
            output_key=output_key,
            beta=beta,
            eps=eps,
            threshold=threshold,
            activation=activation
        )


__all__ = ["F1ScoreCallback"]
