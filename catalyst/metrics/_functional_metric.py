from typing import Callable, Dict

import torch

from catalyst.metrics import ICallbackBatchMetric
from catalyst.metrics._additive import AdditiveValueMetric


class BatchFunctionalMetric(ICallbackBatchMetric):
    """
    Class for custom metric in functional way.
    Note: the loader metrics calculated as average over all batch metrics

    Args:
        metric_fn: metric function, that get outputs, targets and return score as torch.Tensor
        metric_name: metric name
    """

    def __init__(self, metric_fn: Callable, metric_name: str):
        """Init"""
        super().__init__(compute_on_call=True, prefix=metric_name)
        self.metric_fn = metric_fn
        self.cumulative_metric = AdditiveValueMetric()

    def reset(self):
        """Reset all statistics"""
        self.cumulative_metric.reset()

    def update_key_value(
        self, outputs: torch.Tensor, targets: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Calculate metric and update average metric

        Args:
            outputs: tensor of logits
            targets: tensor of targets

        Returns:
            Dict with one element-custom metric
        """
        value = self.update(outputs, targets)
        return {f"{self.prefix}": value}

    def update(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate metric and update average metric

        Args:
            outputs: tensor of logits
            targets: tensor of targets

        Returns:
            custom metric
        """
        value = self.metric_fn(outputs, targets)
        self.cumulative_metric.update(value, len(outputs))
        return value

    def compute(self) -> torch.Tensor:
        """
        Get metric average over all examples

        Returns:
            custom metric
        """
        return self.cumulative_metric.compute()[0]

    def compute_key_value(self) -> Dict[str, torch.Tensor]:
        """
        Get metric average over all examples

        Returns:
            Dict with one element-custom metric
        """
        return {f"{self.prefix}/mean": self.compute()}


__all__ = ["BatchFunctionalMetric"]
