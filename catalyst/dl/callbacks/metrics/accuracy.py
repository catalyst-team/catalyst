from typing import List

from catalyst.core import MultiMetricCallback
from catalyst.dl.callbacks.metrics.functional import get_default_topk_args
from catalyst.utils import metrics


class AccuracyCallback(MultiMetricCallback):
    """Accuracy metric callback.

    It can be used either for
        - multi-class task:
            -you can use accuracy_args.
            -threshold and activation are not required.
            -input_key point on tensor: batch_size.
            -output_key point on tensor: batch_size x num_classes.
        - OR multi-label task, in this case:
            -you must specify threshold and activation.
            -accuracy_args and num_classes will not be used
            (because of there is no method to apply top-k in
            multi-label classification).
            -input_key, output_key point on tensor: batch_size x num_classes.
            -output_key point on a tensor with binary vectors.

    .. note::
        There is no need to choose a type (multi-class/multi label).
        An appropriate type will be chosen automatically via shape of tensors.
    """

    def __init__(
        self,
        input_key: str = "targets",
        output_key: str = "logits",
        prefix: str = "accuracy",
        accuracy_args: List[int] = None,
        num_classes: int = None,
        threshold: float = None,
        activation: str = None,
    ):
        """
        Args:
            input_key (str): input key to use for accuracy calculation;
                specifies our `y_true`
            output_key (str): output key to use for accuracy calculation;
                specifies our `y_pred`
            prefix (str): key for the metric's name
            accuracy_args (List[int]): specifies which accuracy@K to log:
                [1] - accuracy
                [1, 3] - accuracy at 1 and 3
                [1, 3, 5] - accuracy at 1, 3 and 5
            num_classes (int): number of classes to calculate ``accuracy_args``
                if ``accuracy_args`` is None
            threshold (float): threshold for outputs binarization.
            activation (str): An torch.nn activation applied to the outputs.
                Must be one of ``"none"``, ``"Sigmoid"``, or ``"Softmax"``
        """
        list_args = accuracy_args or get_default_topk_args(num_classes)

        super().__init__(
            prefix=prefix,
            metric_fn=metrics.accuracy,
            list_args=list_args,
            input_key=input_key,
            output_key=output_key,
            topk=list_args,
            threshold=threshold,
            activation=activation,
        )


class MapKCallback(MultiMetricCallback):
    """mAP@k metric callback."""

    def __init__(
        self,
        input_key: str = "targets",
        output_key: str = "logits",
        prefix: str = "map",
        map_args: List[int] = None,
        num_classes: int = None,
    ):
        """
        Args:
            input_key (str): input key to use for
                calculation mean average accuracy at k;
                specifies our `y_true`
            output_key (str): output key to use for
                calculation mean average accuracy at k;
                specifies our `y_pred`
            prefix (str): key for the metric's name
            map_args (List[int]): specifies which map@K to log.
                [1] - map@1
                [1, 3] - map@1 and map@3
                [1, 3, 5] - map@1, map@3 and map@5
            num_classes (int): number of classes to calculate ``map_args``
                if ``map_args`` is None
        """
        list_args = map_args or get_default_topk_args(num_classes)

        super().__init__(
            prefix=prefix,
            metric_fn=metrics.mean_average_accuracy,
            list_args=list_args,
            input_key=input_key,
            output_key=output_key,
            topk=list_args,
        )


__all__ = ["AccuracyCallback", "MapKCallback"]
