from typing import List

from catalyst.callbacks.metric import BatchMetricCallback
from catalyst.metrics._hitrate import HitrateMetric
from catalyst.metrics._map import MAPMetric
from catalyst.metrics._mrr import MRRMetric
from catalyst.metrics._ndcg import NDCGMetric


class HitrateCallback(BatchMetricCallback):
    """Hitrate metric callback.
    Computes  HR@topk for the specified values of `topk`.

    Args:
        input_key: input key to use for metric calculation, specifies our `y_pred`
        target_key: output key to use for metric calculation, specifies our `y_true`
        topk_args: specifies which HR@K to log:
            [1] - HR
            [1, 3] - HR at 1 and 3
            [1, 3, 5] - HR at 1, 3 and 5
        log_on_batch: boolean flag to log computed metrics every batch
        prefix: metric prefix
        suffix: metric suffix
    """

    def __init__(
        self,
        input_key: str,
        target_key: str,
        topk_args: List[int] = None,
        log_on_batch: bool = True,
        prefix: str = None,
        suffix: str = None,
    ):
        """Init."""
        super().__init__(
            metric=HitrateMetric(topk_args=topk_args, prefix=prefix, suffix=suffix),
            input_key=input_key,
            target_key=target_key,
            log_on_batch=log_on_batch,
        )


class MAPCallback(BatchMetricCallback):
    """MAP metric callback.
    Computes  MAP@topk for the specified values of `topk`.

    Args:
        input_key: input key to use for metric calculation, specifies our `y_pred`
        target_key: output key to use for metric calculation, specifies our `y_true`
        prefix: key for the metric's name
        topk_args: specifies which MAP@K to log:
            [1] - MAP
            [1, 3] - MAP at 1 and 3
            [1, 3, 5] - MAP at 1, 3 and 5
        log_on_batch: boolean flag to log computed metrics every batch
        prefix: metric prefix
        suffix: metric suffix
    """

    def __init__(
        self,
        input_key: str,
        target_key: str,
        topk_args: List[int] = None,
        log_on_batch: bool = True,
        prefix: str = None,
        suffix: str = None,
    ):
        """Init."""
        super().__init__(
            metric=MAPMetric(topk_args=topk_args, prefix=prefix, suffix=suffix),
            input_key=input_key,
            target_key=target_key,
            log_on_batch=log_on_batch,
        )


class MRRCallback(BatchMetricCallback):
    """MRR metric callback.
    Computes  MRR@topk for the specified values of `topk`.

    Args:
        input_key: input key to use for metric calculation, specifies our `y_pred`
        target_key: output key to use for metric calculation, specifies our `y_true`
        prefix: key for the metric's name
        topk_args: specifies which MRR@K to log:
            [1] - MRR
            [1, 3] - MRR at 1 and 3
            [1, 3, 5] - MRR at 1, 3 and 5
        log_on_batch: boolean flag to log computed metrics every batch
        prefix: metric prefix
        suffix: metric suffix
    """

    def __init__(
        self,
        input_key: str,
        target_key: str,
        topk_args: List[int] = None,
        log_on_batch: bool = True,
        prefix: str = None,
        suffix: str = None,
    ):
        """Init."""
        super().__init__(
            metric=MRRMetric(topk_args=topk_args, prefix=prefix, suffix=suffix),
            input_key=input_key,
            target_key=target_key,
            log_on_batch=log_on_batch,
        )


class NDCGCallback(BatchMetricCallback):
    """NDCG metric callback.
    Computes  NDCG@topk for the specified values of `topk`.

    Args:
        input_key: input key to use for metric calculation, specifies our `y_pred`
        target_key: output key to use for metric calculation, specifies our `y_true`
        prefix: key for the metric's name
        topk_args: specifies which NDCG@K to log:
            [1] - NDCG
            [1, 3] - NDCG at 1 and 3
            [1, 3, 5] - NDCG at 1, 3 and 5
        log_on_batch: boolean flag to log computed metrics every batch
        prefix: metric prefix
        suffix: metric suffix
    """

    def __init__(
        self,
        input_key: str,
        target_key: str,
        topk_args: List[int] = None,
        log_on_batch: bool = True,
        prefix: str = None,
        suffix: str = None,
    ):
        """Init."""
        super().__init__(
            metric=NDCGMetric(topk_args=topk_args, prefix=prefix, suffix=suffix),
            input_key=input_key,
            target_key=target_key,
            log_on_batch=log_on_batch,
        )


__all__ = ["HitrateCallback", "MAPCallback", "MRRCallback", "NDCGCallback"]
