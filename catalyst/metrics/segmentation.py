from typing import Any, Callable, Dict, List, Optional, Tuple
from functools import partial

import torch

from catalyst.metrics.functional.segmentation import (
    _dice,
    _iou,
    _trevsky,
    get_segmentation_statistics,
)
from catalyst.metrics.metric import ICallbackBatchMetric


class RegionBasedMetric(ICallbackBatchMetric):
    """
    Logic class for all region based metrics, like IoU, Dice, Trevsky
    """

    def __init__(
        self,
        metric_fn: Callable,
        compute_on_call: bool = True,
        prefix: Optional[str] = None,
        suffix: Optional[str] = None,
        class_dim: int = 1,
        weights: Optional[List[float]] = None,
        class_names: Optional[List[str]] = None,
        threshold: Optional[float] = 0.5,
    ):
        """

        Args:
            metric_fn: metric function, that get statistics and return score
            compute_on_call: Computes and returns metric value during metric call.
            Used for per-batch logging. default: True
            prefix: metric prefix
            suffix: metric suffix
            class_dim: indicates class dimention (K) for ``outputs`` and
            ``targets`` tensors (default = 1)
            weights: class weights
            class_names: class names
            threshold: threshold for outputs binarization
        """
        super().__init__(compute_on_call, prefix, suffix)
        self.metric_fn = metric_fn
        self.class_dim = class_dim
        self.threshold = threshold
        self.statistics = {}
        self.weights = weights
        self.class_names = class_names
        self._checked_params = False

    def reset(self):
        """Reset all statistics"""
        self.statistics = {}

    def update(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Update segmentation statistics for new data and return intermediate metrics values.

        Args:
            outputs: tensor of logits
            targets: tensor of targets

        Returns:
            metric for each class
        """
        tp, fp, fn = get_segmentation_statistics(
            outputs=outputs.cpu().detach(),
            targets=targets.cpu().detach(),
            class_dim=self.class_dim,
            threshold=self.threshold,
        )

        for idx, (tp_class, fp_class, fn_class) in enumerate(zip(tp, fp, fn), start=1):
            if idx in self.statistics:
                self.statistics[idx]["tp"] += tp_class
                self.statistics[idx]["fp"] += fp_class
                self.statistics[idx]["fn"] += fn_class
            else:
                self.statistics[idx] = {}
                self.statistics[idx]["tp"] = tp_class
                self.statistics[idx]["fp"] = fp_class
                self.statistics[idx]["fn"] = fn_class

        values = self.metric_fn(tp, fp, fn)
        return values

    def _check_parameters(self):
        # check class_names
        if self.class_names is not None:
            assert len(self.class_names) == len(self.statistics), (
                f"the number of class names must be equal to the number of classes, got weights"
                f" {len(self.class_names)} and classes: {len(self.statistics)}"
            )
        else:
            self.class_names = [f"class_{idx}" for idx in range(1, len(self.statistics) + 1)]
        if self.weights is not None:
            assert len(self.weights) == len(self.statistics), (
                f"the number of weights must be equal to the number of classes, got weights"
                f" {len(self.weights)} and classes: {len(self.statistics)}"
            )

    def update_key_value(
        self, outputs: torch.Tensor, targets: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Update segmentation statistics for new data and return intermediate metrics values.

        Args:
            outputs: tensor of logits
            targets: tensor of targets

        Returns:
            dict of metric for each class and weighted (if weights were given) metric
        """
        values = self.update(outputs, targets)
        # need only one time
        if not self._checked_params:
            self._check_parameters()
            self._checked_params = True
        metrics = {
            f"{self.prefix}/{self.class_names[idx-1]}": value
            for idx, value in enumerate(values, start=1)
        }
        if self.weights is not None:
            weighted_metric = 0
            for idx, value in enumerate(values):
                weighted_metric += value * self.weights[idx]
            metrics[f"{self.prefix}/weighted"] = weighted_metric
        # convert torch.Tensor to float
        metrics = {k: float(v) for k, v in metrics.items()}
        return metrics

    def compute_key_value(self) -> Dict[str, torch.Tensor]:
        """
        Compute segmentation metric for all data and return results in key-value format

        Returns:
             dict of metrics, including micro, macro and weighted (if weights were given) metrics
        """
        metrics = {}
        total_statistics = {}
        micro_metric = 0
        if self.weights is not None:
            weighted_metric = 0
        for class_idx, statistics in self.statistics.items():
            value = self.metric_fn(**statistics)
            micro_metric += value
            if self.weights is not None:
                weighted_metric += value * self.weights[class_idx - 1]
            metrics[f"{self.prefix}/{self.class_names[class_idx-1]}"] = value
            for stats_name, value in statistics.items():
                total_statistics[stats_name] = total_statistics.get(stats_name, 0) + value
        micro_metric /= len(self.statistics)
        macro_metric = self.metric_fn(**total_statistics)
        metrics[f"{self.prefix}/micro"] = micro_metric
        metrics[f"{self.prefix}/macro"] = macro_metric
        if self.weights is not None:
            metrics[f"{self.prefix}/weighted"] = weighted_metric
        # convert torch.Tensor to float
        metrics = {k: float(v) for k, v in metrics.items()}
        return metrics

    def compute(self):
        return self.compute_key_value()


class IOUMetric(RegionBasedMetric):
    """
    IoU Metric
    iou score = intersection / union = tp / (tp + fp + fn)
    """

    def __init__(
        self,
        compute_on_call: bool = True,
        prefix: Optional[str] = "iou",
        suffix: Optional[str] = None,
        class_dim: int = 1,
        weights: Optional[List[float]] = None,
        class_names: Optional[List[str]] = None,
        threshold: Optional[float] = None,
        eps: float = 1e-7,
    ):
        """

        Args:
            compute_on_call: Computes and returns metric value during metric call.
            Used for per-batch logging. default: True
            prefix: metric prefix
            suffix: metric suffix
            class_dim: indicates class dimention (K) for ``outputs`` and
            ``targets`` tensors (default = 1)
            weights: class weights
            class_names: class names
            threshold: threshold for outputs binarization
            eps: epsilon to avoid zero division
        """
        metric_fn = partial(_iou, eps=eps)
        super().__init__(
            metric_fn=metric_fn,
            compute_on_call=compute_on_call,
            prefix=prefix,
            suffix=suffix,
            class_dim=class_dim,
            weights=weights,
            class_names=class_names,
            threshold=threshold,
        )


class DiceMetric(RegionBasedMetric):
    """
    Dice Metric
    dice score = 2 * intersection / (intersection + union)) = \
    = 2 * tp / (2 * tp + fp + fn)
    """

    def __init__(
        self,
        compute_on_call: bool = True,
        prefix: Optional[str] = "dice",
        suffix: Optional[str] = None,
        class_dim: int = 1,
        weights: Optional[List[float]] = None,
        class_names: Optional[List[str]] = None,
        threshold: Optional[float] = None,
        eps: float = 1e-7,
    ):
        """

        Args:
            compute_on_call: Computes and returns metric value during metric call.
            Used for per-batch logging. default: True
            prefix: metric prefix
            suffix: metric suffix
            class_dim: indicates class dimention (K) for ``outputs`` and
            ``targets`` tensors (default = 1)
            weights: class weights
            class_names: class names
            threshold: threshold for outputs binarization
            eps: epsilon to avoid zero division
        """
        metric_fn = partial(_dice, eps=eps)
        super().__init__(
            metric_fn=metric_fn,
            compute_on_call=compute_on_call,
            prefix=prefix,
            suffix=suffix,
            class_dim=class_dim,
            weights=weights,
            class_names=class_names,
            threshold=threshold,
        )


class TrevskyMetric(RegionBasedMetric):
    """
    Trevsky Metric
    trevsky score = tp / (tp + fp * beta + fn * alpha)
    """

    def __init__(
        self,
        alpha: float,
        beta: Optional[float] = None,
        compute_on_call: bool = True,
        prefix: Optional[str] = "trevsky_index",
        suffix: Optional[str] = None,
        class_dim: int = 1,
        weights: Optional[List[float]] = None,
        class_names: Optional[List[str]] = None,
        threshold: Optional[float] = None,
        eps: float = 1e-7,
    ):
        """

        Args:
            alpha: false negative coefficient, bigger alpha bigger penalty for
            false negative. if beta is None, alpha must be in (0, 1)
            beta: false positive coefficient, bigger alpha bigger penalty for false
            positive. Must be in (0, 1), if None beta = (1 - alpha)
            compute_on_call: Computes and returns metric value during metric call.
            Used for per-batch logging. default: True
            prefix: metric prefix
            suffix: metric suffix
            class_dim: indicates class dimension (K) for ``outputs`` and
            ``targets`` tensors (default = 1)
            weights: class weights
            class_names: class names
            threshold: threshold for outputs binarization
            eps: epsilon to avoid zero division
        """
        if beta is None:
            assert 0 < alpha < 1, "if beta=None, alpha must be in (0, 1)"
            beta = 1 - alpha
        metric_fn = partial(_trevsky, alpha=alpha, beta=beta, eps=eps)
        super().__init__(
            metric_fn=metric_fn,
            compute_on_call=compute_on_call,
            prefix=prefix,
            suffix=suffix,
            class_dim=class_dim,
            weights=weights,
            class_names=class_names,
            threshold=threshold,
        )


JaccardMetric = IOUMetric

__all__ = [
    "RegionBasedMetric",
    "IOUMetric",
    "JaccardMetric",
    "DiceMetric",
    "TrevskyMetric",
]
