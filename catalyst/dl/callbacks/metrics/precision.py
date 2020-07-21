from typing import List

from catalyst.core import MultiMetricCallback
from catalyst.core.runner import IRunner
from catalyst.dl.callbacks.metrics.functional import get_default_topk_args
from catalyst.utils import metrics


class AveragePrecisionCallback(MultiMetricCallback):
    """AveragePrecision@k metric callback."""

    def __init__(
        self,
        input_key: str = "targets",
        output_key: str = "logits",
        prefix: str = "average_precision",
        ap_args: List[int] = None,
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
            ap_args (List[int]): specifies which ap@K to log.
                [1] - ap@1
                [1, 3] - ap@1 and map@3
                [1, 3, 5] - ap@1, map@3 and map@5
            num_classes (int): number of classes to calculate ``ap_args``
                if ``ap_args`` is None
        """
        list_args = ap_args or get_default_topk_args(num_classes)

        super().__init__(
            prefix=prefix,
            metric_fn=metrics.average_precision,
            list_args=list_args,
            input_key=input_key,
            output_key=output_key,
            topk=list_args,
        )

    def on_batch_end(self, runner: IRunner) -> None:
        """Batch end hook.

        Args:
            runner (IRunner): current runner
        """
        topk_metrics = self._compute_metric(runner)

        for arg, per_class_metrics in zip(self.list_args, topk_metrics):
            if isinstance(arg, int):
                prefix = f"{self.prefix}{arg:02}"
            else:
                prefix = f"{self.prefix}_{arg}"

            for i, class_metric in enumerate(per_class_metrics):
                key = f"{prefix}/class_{i:02}"
                runner.batch_metrics[key] = class_metric * self.multiplier


class MeanAveragePrecisionCallback(MultiMetricCallback):
    """MeanAveragePrecision@k metric callback."""

    def __init__(
        self,
        input_key: str = "targets",
        output_key: str = "logits",
        prefix: str = "mean_average_precision",
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
            metric_fn=metrics.mean_average_precision,
            list_args=list_args,
            input_key=input_key,
            output_key=output_key,
            topk=list_args,
        )


__all__ = ["AveragePrecisionCallback", "MeanAveragePrecisionCallback"]
