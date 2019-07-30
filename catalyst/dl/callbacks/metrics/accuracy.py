from typing import List

from catalyst.dl.core import MultiMetricCallback
from catalyst.dl.utils import criterion


def _get_default_accuracy_args(num_classes: int) -> List[int]:
    """
    Calculate list params for Accuracy@k and mAP@k
    Args:
        num_classes (int): number of classes
    """
    result = [1]

    if num_classes is None:
        return result

    if num_classes >= 3:
        result.append(3)
    if num_classes >= 5:
        result.append(5)

    return result


class AccuracyCallback(MultiMetricCallback):
    """
    Accuracy metric callback.
    """

    def __init__(
        self,
        input_key: str = "targets",
        output_key: str = "logits",
        prefix: str = "accuracy",
        accuracy_args: List[int] = None,
        num_classes: int = None,
    ):
        """
        Args:
            input_key (str): input key to use for accuracy calculation;
                specifies our `y_true`.
            output_key (str): output key to use for accuracy calculation;
                specifies our `y_pred`.
            prefix (str): key for the metric's name
            accuracy_args (List[int]): specifies which accuracy@K to log.
                [1] - accuracy
                [1, 3] - accuracy at 1 and 3
                [1, 3, 5] - accuracy at 1, 3 and 5
            num_classes (int): number of classes to calculate ``accuracy_args``
                if ``accuracy_args`` is None
        """
        list_args = accuracy_args or _get_default_accuracy_args(num_classes)

        super().__init__(
            prefix=prefix,
            metric_fn=criterion.accuracy,
            list_args=list_args,
            input_key=input_key,
            output_key=output_key
        )


class MapKCallback(MultiMetricCallback):
    """
    mAP@k metric callback.
    """

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
                specifies our `y_true`.
            output_key (str): output key to use for
                calculation mean average accuracy at k;
                specifies our `y_pred`.
            prefix (str): key for the metric's name
            map_args (List[int]): specifies which map@K to log.
                [1] - map@1
                [1, 3] - map@1 and map@3
                [1, 3, 5] - map@1, map@3 and map@5
            num_classes (int): number of classes to calculate ``map_args``
                if ``map_args`` is None
        """
        list_args = map_args or _get_default_accuracy_args(num_classes)

        super().__init__(
            prefix=prefix,
            metric_fn=criterion.mean_average_accuracy,
            list_args=list_args,
            input_key=input_key,
            output_key=output_key
        )


__all__ = ["AccuracyCallback", "MapKCallback"]
