from typing import List

from catalyst.dl.core import Callback, MetricCallback, MultiMetricCallback
from catalyst.dl.utils import criterion



class IouCallback(MetricCallback):
    """
    IoU (Jaccard) metric callback.
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
            input_key (str): input key to use for iou calculation
                specifies our ``y_true``.
            output_key (str): output key to use for iou calculation;
                specifies our ``y_pred``
            prefix (str): key to store in logs
            eps (float): epsilon to avoid zero division
            threshold (float): threshold for outputs binarization
            activation (str): An torch.nn activation applied to the outputs.
                Must be one of ['none', 'Sigmoid', 'Softmax2d']
        """
        super().__init__(
            prefix=prefix,
            metric_fn=criterion.iou,
            input_key=input_key,
            output_key=output_key,
            eps=eps,
            threshold=threshold,
            activation=activation
        )


JaccardCallback = IouCallback



def _get_default_classwise_iou_args(num_classes: int) -> List[int]:
    return [str(i) for i in range(num_classes)]


class ClasswiseIouCallback(MultiMetricCallback):
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
        assert classes is not None or num_classes is not None, \
            "You should specify either 'classes' or 'num_classes'"
        list_args = classes or _get_default_classwise_iou_args(num_classes)

        super().__init__(
            prefix=prefix,
            metric_fn=criterion.iou,
            list_args=list_args,
            input_key=input_key,
            output_key=output_key,
            eps=eps,
            threshold=threshold,
            activation=activation,
        )


ClasswiseJaccardCallback = ClasswiseIouCallback



__all__ = [
    "IouCallback", "JaccardCallback",
    "ClasswiseIouCallback", "ClasswiseJaccardCallback"
]
