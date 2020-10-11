from typing import List

from catalyst.callbacks.metric import BatchMetricCallback
from catalyst.metrics.iou import iou


def _get_default_classwise_iou_args(num_classes: int) -> List[int]:
    assert num_classes > 0, "num_classes must be greater than 0"
    return [str(i) for i in range(num_classes)]


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
            Must be one of ``'none'``, ``'Sigmoid'``, ``'Softmax2d'``
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
                Must be one of ``'none'``, ``'Sigmoid'``, ``'Softmax2d'``
        """
        super().__init__(
            prefix=prefix,
            metric_fn=iou,
            input_key=input_key,
            output_key=output_key,
            eps=eps,
            threshold=threshold,
            activation=activation,
        )


JaccardCallback = IouCallback


class ClasswiseIouCallback(BatchMetricCallback):
    """Classwise IoU (Jaccard) metric callback."""

    def __init__(
        self,
        input_key: str = "targets",
        output_key: str = "logits",
        prefix: str = "iou",
        classes: List[str] = None,
        num_classes: int = None,
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
            prefix: key to store in logs (will be prefix_class_name)
            classes: list of class names
                You should specify either 'classes' or 'num_classes'
            num_classes: number of classes
                You should specify either 'classes' or 'num_classes'
            eps: epsilon to avoid zero division
            threshold: threshold for outputs binarization
            activation: An torch.nn activation applied to the outputs.
                Must be one of ``'none'``, ``'Sigmoid'``, ``'Softmax2d'``
        """
        assert (
            classes is not None or num_classes is not None
        ), "You should specify either 'classes' or 'num_classes'"
        list_args = classes or _get_default_classwise_iou_args(num_classes)

        super().__init__(
            prefix=prefix,
            metric_fn=iou,
            list_args=list_args,
            input_key=input_key,
            output_key=output_key,
            classes=list_args,
            eps=eps,
            threshold=threshold,
            activation=activation,
        )


ClasswiseJaccardCallback = ClasswiseIouCallback

__all__ = [
    "IouCallback",
    "JaccardCallback",
    "ClasswiseIouCallback",
    "ClasswiseJaccardCallback",
]
