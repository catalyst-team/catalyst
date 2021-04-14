from typing import Callable, Dict, List, Optional
from functools import partial

import torch

from catalyst.metrics._metric import ICallbackBatchMetric
from catalyst.metrics.functional._segmentation import (
    _dice,
    _iou,
    _trevsky,
    get_segmentation_statistics,
)
from catalyst.utils.distributed import all_gather, get_rank


class RegionBasedMetric(ICallbackBatchMetric):
    """Logic class for all region based metrics, like IoU, Dice, Trevsky.

    Args:
        metric_fn: metric function, that get statistics and return score
        metric_name: name of the metric
        class_dim: indicates class dimension (K) for ``outputs`` and
            ``targets`` tensors (default = 1)
        weights: class weights
        class_names: class names
        threshold: threshold for outputs binarization
        compute_on_call: Computes and returns metric value during metric call.
            Used for per-batch logging. default: True
        prefix: metric prefix
        suffix: metric suffix
    """

    def __init__(
        self,
        metric_fn: Callable,
        metric_name: str,
        class_dim: int = 1,
        weights: Optional[List[float]] = None,
        class_names: Optional[List[str]] = None,
        threshold: Optional[float] = 0.5,
        compute_on_call: bool = True,
        prefix: Optional[str] = None,
        suffix: Optional[str] = None,
    ):
        """Init"""
        super().__init__(compute_on_call, prefix, suffix)
        self.metric_fn = metric_fn
        self.metric_name = metric_name
        self.class_dim = class_dim
        self.threshold = threshold
        # statistics = {class_idx: {"tp":, "fn": , "fp": "tn": }}
        self.statistics = {}
        self.weights = weights
        self.class_names = class_names
        self._checked_params = False
        self._is_ddp = False

    def _check_parameters(self):
        # check class_names
        if self.class_names is not None:
            assert len(self.class_names) == len(self.statistics), (
                f"the number of class names must be equal to the number of classes, got weights"
                f" {len(self.class_names)} and classes: {len(self.statistics)}"
            )
        else:
            self.class_names = [f"class_{idx:02d}" for idx in range(len(self.statistics))]
        if self.weights is not None:
            assert len(self.weights) == len(self.statistics), (
                f"the number of weights must be equal to the number of classes, got weights"
                f" {len(self.weights)} and classes: {len(self.statistics)}"
            )

    def reset(self):
        """Reset all statistics"""
        self.statistics = {}
        self._is_ddp = get_rank() > -1

    def update(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Updates segmentation statistics with new data and return intermediate metrics values.

        Args:
            outputs: tensor of logits
            targets: tensor of targets

        Returns:
            metric for each class
        """
        tp, fp, fn = get_segmentation_statistics(
            outputs=outputs.detach(),
            targets=targets.detach(),
            class_dim=self.class_dim,
            threshold=self.threshold,
        )

        for idx, (tp_class, fp_class, fn_class) in enumerate(zip(tp, fp, fn)):
            if idx in self.statistics:
                self.statistics[idx]["tp"] += tp_class
                self.statistics[idx]["fp"] += fp_class
                self.statistics[idx]["fn"] += fn_class
            else:
                self.statistics[idx] = {}
                self.statistics[idx]["tp"] = tp_class
                self.statistics[idx]["fp"] = fp_class
                self.statistics[idx]["fn"] = fn_class

        metrics_per_class = self.metric_fn(tp, fp, fn)
        return metrics_per_class

    def update_key_value(
        self, outputs: torch.Tensor, targets: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Updates segmentation statistics with new data and return intermediate metrics values.

        Args:
            outputs: tensor of logits
            targets: tensor of targets

        Returns:
            dict of metric for each class and weighted (if weights were given) metric
        """
        metrics_per_class = self.update(outputs, targets)
        macro_metric = torch.mean(metrics_per_class)
        # need only one time
        if not self._checked_params:
            self._check_parameters()
            self._checked_params = True
        metrics = {
            f"{self.prefix}{self.metric_name}{self.suffix}/{self.class_names[idx]}": value
            for idx, value in enumerate(metrics_per_class)
        }
        metrics[f"{self.prefix}{self.metric_name}{self.suffix}"] = macro_metric
        if self.weights is not None:
            weighted_metric = 0
            for idx, value in enumerate(metrics_per_class):
                weighted_metric += value * self.weights[idx]
            metrics[f"{self.prefix}{self.metric_name}{self.suffix}/_weighted"] = weighted_metric
        # convert torch.Tensor to float
        # metrics = {k: float(v) for k, v in metrics.items()}
        return metrics

    def compute_key_value(self) -> Dict[str, torch.Tensor]:
        """
        Compute segmentation metric for all data and return results in key-value format

        Returns:
             dict of metrics, including micro, macro and weighted (if weights were given) metrics
        """
        metrics = {}
        total_statistics = {}
        macro_metric = 0
        weighted_metric = 0

        # @TODO: ddp hotfix, could be done better
        if self._is_ddp:
            for _, statistics in self.statistics.items():
                for key in statistics:
                    device = statistics[key].device
                    value: List[torch.Tensor] = all_gather(statistics[key].cpu())
                    value: torch.Tensor = torch.sum(torch.vstack(value), dim=0).to(device)
                    statistics[key] = value

        for class_idx, statistics in self.statistics.items():
            value = self.metric_fn(**statistics)
            macro_metric += value
            if self.weights is not None:
                weighted_metric += value * self.weights[class_idx]
            metrics[
                f"{self.prefix}{self.metric_name}{self.suffix}/{self.class_names[class_idx]}"
            ] = value
            for stats_name, value in statistics.items():
                total_statistics[stats_name] = total_statistics.get(stats_name, 0) + value
        macro_metric /= len(self.statistics)
        micro_metric = self.metric_fn(**total_statistics)
        metrics[f"{self.prefix}{self.metric_name}{self.suffix}/_micro"] = micro_metric
        metrics[f"{self.prefix}{self.metric_name}{self.suffix}"] = macro_metric
        metrics[f"{self.prefix}{self.metric_name}{self.suffix}/_macro"] = macro_metric
        if self.weights is not None:
            metrics[f"{self.prefix}{self.metric_name}{self.suffix}/_weighted"] = weighted_metric
        # convert torch.Tensor to float
        # metrics = {k: float(v) for k, v in metrics.items()}
        return metrics

    def compute(self):
        """@TODO: Docs."""
        return self.compute_key_value()


class IOUMetric(RegionBasedMetric):
    """
    IoU Metric,
    iou score = intersection / union = tp / (tp + fp + fn).

    Args:
        class_dim: indicates class dimension (K) for ``outputs`` and
            ``targets`` tensors (default = 1)
        weights: class weights
        class_names: class names
        threshold: threshold for outputs binarization
        eps: epsilon to avoid zero division
        compute_on_call: Computes and returns metric value during metric call.
            Used for per-batch logging. default: True
        prefix: metric prefix
        suffix: metric suffix
    """

    def __init__(
        self,
        class_dim: int = 1,
        weights: Optional[List[float]] = None,
        class_names: Optional[List[str]] = None,
        threshold: Optional[float] = None,
        eps: float = 1e-7,
        compute_on_call: bool = True,
        prefix: Optional[str] = None,
        suffix: Optional[str] = None,
    ):
        """Init."""
        metric_fn = partial(_iou, eps=eps)
        super().__init__(
            metric_fn=metric_fn,
            metric_name="iou",
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
    Dice Metric,
    dice score = 2 * intersection / (intersection + union)) = 2 * tp / (2 * tp + fp + fn)

    Args:
        class_dim: indicates class dimention (K) for ``outputs`` and
        ``targets`` tensors (default = 1)
        weights: class weights
        class_names: class names
        threshold: threshold for outputs binarization
        eps: epsilon to avoid zero division
        compute_on_call: Computes and returns metric value during metric call.
            Used for per-batch logging. default: True
        prefix: metric prefix
        suffix: metric suffix
    """

    def __init__(
        self,
        class_dim: int = 1,
        weights: Optional[List[float]] = None,
        class_names: Optional[List[str]] = None,
        threshold: Optional[float] = None,
        eps: float = 1e-7,
        compute_on_call: bool = True,
        prefix: Optional[str] = None,
        suffix: Optional[str] = None,
    ):
        """Init."""
        metric_fn = partial(_dice, eps=eps)
        super().__init__(
            metric_fn=metric_fn,
            metric_name="dice",
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
    Trevsky Metric,
    trevsky score = tp / (tp + fp * beta + fn * alpha)

    Args:
        alpha: false negative coefficient, bigger alpha bigger penalty for
            false negative. if beta is None, alpha must be in (0, 1)
        beta: false positive coefficient, bigger alpha bigger penalty for false
            positive. Must be in (0, 1), if None beta = (1 - alpha)
        class_dim: indicates class dimension (K) for ``outputs`` and
            ``targets`` tensors (default = 1)
        weights: class weights
        class_names: class names
        threshold: threshold for outputs binarization
        eps: epsilon to avoid zero division
        compute_on_call: Computes and returns metric value during metric call.
            Used for per-batch logging. default: True
        prefix: metric prefix
        suffix: metric suffix
    """

    def __init__(
        self,
        alpha: float,
        beta: Optional[float] = None,
        class_dim: int = 1,
        weights: Optional[List[float]] = None,
        class_names: Optional[List[str]] = None,
        threshold: Optional[float] = None,
        eps: float = 1e-7,
        compute_on_call: bool = True,
        prefix: Optional[str] = None,
        suffix: Optional[str] = None,
    ):
        """Init."""
        if beta is None:
            assert 0 < alpha < 1, "if beta=None, alpha must be in (0, 1)"
            beta = 1 - alpha
        metric_fn = partial(_trevsky, alpha=alpha, beta=beta, eps=eps)
        super().__init__(
            metric_fn=metric_fn,
            metric_name="trevsky",
            compute_on_call=compute_on_call,
            prefix=prefix,
            suffix=suffix,
            class_dim=class_dim,
            weights=weights,
            class_names=class_names,
            threshold=threshold,
        )


__all__ = [
    "RegionBasedMetric",
    "IOUMetric",
    "DiceMetric",
    "TrevskyMetric",
]
