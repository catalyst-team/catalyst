from typing import List, Union

import torch

from catalyst.callbacks.metric import BatchMetricCallback
from catalyst.metrics._accuracy import AccuracyMetric, MultilabelAccuracyMetric


class AccuracyCallback(BatchMetricCallback):
    """Accuracy metric callback.
    Computes multiclass accuracy@topk for the specified values of `topk`.

    Args:
        input_key: input key to use for metric calculation, specifies our `y_pred`
        target_key: output key to use for metric calculation, specifies our `y_true`
        topk_args: specifies which accuracy@K to log:
            [1] - accuracy
            [1, 3] - accuracy at 1 and 3
            [1, 3, 5] - accuracy at 1, 3 and 5
        num_classes: number of classes to calculate ``topk_args`` if ``accuracy_args`` is None
        log_on_batch: boolean flag to log computed metrics every batch
        prefix: metric prefix
        suffix: metric suffix
    """

    def __init__(
        self,
        input_key: str,
        target_key: str,
        topk_args: List[int] = None,
        num_classes: int = None,
        log_on_batch: bool = True,
        prefix: str = None,
        suffix: str = None,
    ):
        """Init."""
        super().__init__(
            metric=AccuracyMetric(
                topk_args=topk_args, num_classes=num_classes, prefix=prefix, suffix=suffix
            ),
            input_key=input_key,
            target_key=target_key,
            log_on_batch=log_on_batch,
        )


class MultilabelAccuracyCallback(BatchMetricCallback):
    """Multilabel accuracy metric callback.
    Computes multilabel accuracy@topk for the specified values of `topk`.

    Args:
        input_key: input key to use for metric calculation, specifies our `y_pred`
        target_key: output key to use for metric calculation, specifies our `y_true`
        threshold: thresholds for model scores
        log_on_batch: boolean flag to log computed metrics every batch
        prefix: metric prefix
        suffix: metric suffix
    """

    def __init__(
        self,
        input_key: str,
        target_key: str,
        threshold: Union[float, torch.Tensor] = 0.5,
        log_on_batch: bool = True,
        prefix: str = None,
        suffix: str = None,
    ):
        """Init."""
        super().__init__(
            metric=MultilabelAccuracyMetric(threshold=threshold, prefix=prefix, suffix=suffix),
            input_key=input_key,
            target_key=target_key,
            log_on_batch=log_on_batch,
        )


__all__ = ["AccuracyCallback", "MultilabelAccuracyCallback"]
