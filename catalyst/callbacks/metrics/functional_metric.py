from typing import Callable, Dict, Iterable, Union

from catalyst.callbacks.metric import FunctionalBatchMetricCallback
from catalyst.metrics._functional_metric import FunctionalBatchMetric


class FunctionalMetricCallback(FunctionalBatchMetricCallback):
    """

    Args:
        input_key: input key to use for metric calculation, specifies our `y_pred`
        target_key: output key to use for metric calculation, specifies our `y_true`
        metric_fn: metric function, that get outputs, targets and return score as torch.Tensor
        metric_key: key to store computed metric in ``runner.batch_metrics`` dictionary
        compute_on_call: Computes and returns metric value during metric call.
            Used for per-batch logging. default: True
        log_on_batch: boolean flag to log computed metrics every batch
        prefix: metric prefix
        suffix: metric suffix
    """

    def __init__(
        self,
        input_key: Union[str, Iterable[str], Dict[str, str]],
        target_key: Union[str, Iterable[str], Dict[str, str]],
        metric_fn: Callable,
        metric_key: str,
        compute_on_call: bool = True,
        log_on_batch: bool = True,
        prefix: str = None,
        suffix: str = None,
    ):
        """Init."""
        super().__init__(
            metric=FunctionalBatchMetric(
                metric_fn=metric_fn,
                metric_key=metric_key,
                compute_on_call=compute_on_call,
                prefix=prefix,
                suffix=suffix,
            ),
            input_key=input_key,
            target_key=target_key,
            log_on_batch=log_on_batch,
        )


__all__ = ["FunctionalMetricCallback"]
