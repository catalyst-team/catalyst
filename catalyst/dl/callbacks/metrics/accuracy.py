from typing import List

from catalyst.dl.core import MultiMetricCallback
from catalyst.dl.utils import criterion


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
    ):
        """
        Args:
            input_key: input key to use for accuracy calculation;
                specifies our `y_true`.
            output_key: output key to use for accuracy calculation;
                specifies our `y_pred`.
            accuracy_args: specifies which accuracy@K to log.
                [1] - accuracy
                [1, 3] - accuracy at 1 and 3
                [1, 3, 5] - accuracy at 1, 3 and 5
        """
        super().__init__(
            prefix=prefix,
            metric_fn=criterion.accuracy,
            list_args=accuracy_args or [1],
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
    ):
        """
        Args:
            input_key: input key to use for
                calculation mean average accuracy at k;
                specifies our `y_true`.
            output_key: output key to use for
                calculation mean average accuracy at k;
                specifies our `y_pred`.
            map_args: specifies which map@K to log.
                [1] - map@1
                [1, 3] - map@1 and map@3
                [1, 3, 5] - map@1, map@3 and map@5
        """
        super().__init__(
            prefix=prefix,
            metric_fn=criterion.mean_average_accuracy,
            list_args=map_args or [1],
            input_key=input_key,
            output_key=output_key
        )


__all__ = ["AccuracyCallback", "MapKCallback"]
