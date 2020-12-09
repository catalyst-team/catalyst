from catalyst.callbacks.metric import BatchMetricCallback
from catalyst.metrics.functional import wrap_metric_fn_with_activation
from catalyst.metrics.iou import iou


class IouCallback(BatchMetricCallback):
    """IoU (Jaccard) metric callback.

    Args:
        input_key: input key to use for iou calculation
            specifies our ``y_true``
        output_key: output key to use for iou calculation;
            specifies our ``y_pred``
        prefix: key to store in logs
        eps: epsilon to avoid zero division
        threshold: threshold for outputs binarization
        activation: An torch.nn activation applied to the outputs.
            Must be one of ``'none'``, ``'Sigmoid'``, ``'Softmax'``
    """

    def __init__(
        self,
        input_key: str = "targets",
        output_key: str = "logits",
        prefix: str = "iou",
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
            eps: epsilon to avoid zero division
            threshold: threshold for outputs binarization
            activation: An torch.nn activation applied to the outputs.
                Must be one of ``'none'``, ``'Sigmoid'``, ``'Softmax'``
        """
        super().__init__(
            prefix=prefix,
            metric_fn=wrap_metric_fn_with_activation(
                metric_fn=iou, activation=activation
            ),
            input_key=input_key,
            output_key=output_key,
            eps=eps,
            threshold=threshold,
        )


JaccardCallback = IouCallback


__all__ = [
    "IouCallback",
    "JaccardCallback",
]
