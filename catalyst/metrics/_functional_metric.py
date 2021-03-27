from typing import Callable, Dict

import torch

from catalyst.metrics import ICallbackBatchMetric
from catalyst.metrics._additive import AdditiveValueMetric


class FunctionalBatchMetric(ICallbackBatchMetric):
    """Class for custom metrics in a functional way.

    Args:
        metric_fn: metric function, that get outputs, targets and return score as torch.Tensor
        metric_key: metric name
        compute_on_call: Computes and returns metric value during metric call.
            Used for per-batch logging. default: True
        prefix: metric prefix
        suffix: metric suffix

    .. note::

        Loader metrics calculated as average over all batch metrics.

    """

    def __init__(
        self,
        metric_fn: Callable,
        metric_key: str,
        compute_on_call: bool = True,
        prefix: str = None,
        suffix: str = None,
    ):
        """Init"""
        super().__init__(compute_on_call=compute_on_call, prefix=prefix, suffix=suffix)
        self.metric_fn = metric_fn
        self.metric_name = f"{self.prefix}{metric_key}{self.suffix}"
        self.additive_metric = AdditiveValueMetric()

    def reset(self):
        """Reset all statistics"""
        self.additive_metric.reset()

    def update(self, batch_size: int, *args, **kwargs) -> torch.Tensor:
        """
        Calculate metric and update average metric

        Args:
            batch_size: current batch size for metric statistics aggregation
            *args: args for metric_fn
            **kwargs: kwargs for metric_fn

        Returns:
            custom metric
        """
        value = self.metric_fn(*args, **kwargs)
        self.additive_metric.update(float(value), batch_size)
        return value

    def update_key_value(self, batch_size: int, *args, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Calculate metric and update average metric

        Args:
            batch_size: current batch size for metric statistics aggregation
            *args: args for metric_fn
            **kwargs: kwargs for metric_fn

        Returns:
            Dict with one element-custom metric
        """
        value = self.update(batch_size, *args, **kwargs)
        return {f"{self.metric_name}": value}

    def compute(self) -> torch.Tensor:
        """
        Get metric average over all examples

        Returns:
            custom metric
        """
        return self.additive_metric.compute()

    def compute_key_value(self) -> Dict[str, torch.Tensor]:
        """
        Get metric average over all examples

        Returns:
            Dict with one element-custom metric
        """
        mean, std = self.compute()
        return {
            self.metric_name: mean,
            f"{self.metric_name}/mean": mean,
            f"{self.metric_name}/std": std,
        }


__all__ = ["FunctionalBatchMetric"]
