from typing import Any, Callable, Dict, Iterable, List

import torch

from catalyst.metrics._additive import AdditiveMetric
from catalyst.metrics._metric import ICallbackBatchMetric


class TopKMetric(ICallbackBatchMetric):
    """
    Base class for `topk` metrics.

    Args:
        metric_name: name of the metric
        metric_function: metric calculation function
        topk: list of `topk` for metric@topk computing
        compute_on_call: if True, computes and returns metric value during metric call
        prefix: metric prefix
        suffix: metric suffix
    """

    def __init__(
        self,
        metric_name: str,
        metric_function: Callable,
        topk: Iterable[int] = None,
        compute_on_call: bool = True,
        prefix: str = None,
        suffix: str = None,
    ):
        """Init TopKMetric"""
        super().__init__(compute_on_call=compute_on_call, prefix=prefix, suffix=suffix)
        self.metric_name = metric_name
        self.metric_function = metric_function
        self.topk = topk or (1,)
        self.metrics: List[AdditiveMetric] = [
            AdditiveMetric() for _ in range(len(self.topk))
        ]

    def reset(self) -> None:
        """Reset all fields"""
        for metric in self.metrics:
            metric.reset()

    def update(self, logits: torch.Tensor, targets: torch.Tensor) -> List[float]:
        """
        Update metric value with value for new data
        and return intermediate metrics values.

        Args:
            logits (torch.Tensor): tensor of logits
            targets (torch.Tensor): tensor of targets

        Returns:
            list of metric@k values
        """
        values = self.metric_function(logits, targets, topk=self.topk)
        values = [v.item() for v in values]
        for value, metric in zip(values, self.metrics):
            metric.update(value, len(targets))
        return values

    def update_key_value(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> Dict[str, float]:
        """
        Update metric value with value for new data and return intermediate metrics
        values in key-value format.

        Args:
            logits (torch.Tensor): tensor of logits
            targets (torch.Tensor): tensor of targets

        Returns:
            dict of metric@k values
        """
        values = self.update(logits=logits, targets=targets)
        output = {
            f"{self.prefix}{self.metric_name}{key:02d}{self.suffix}": value
            for key, value in zip(self.topk, values)
        }
        return output

    def compute(self) -> Any:
        """
        Compute metric for all data

        Returns:
            list of mean values, list of std values
        """
        means, stds = zip(*(metric.compute() for metric in self.metrics))
        return means, stds

    def compute_key_value(self) -> Dict[str, float]:
        """
        Compute metric for all data and return results in key-value format

        Returns:
            dict of metrics
        """
        means, stds = self.compute()
        output_mean = {
            f"{self.prefix}{self.metric_name}{key:02d}{self.suffix}": value
            for key, value in zip(self.topk, means)
        }
        output_std = {
            f"{self.prefix}{self.metric_name}{key:02d}{self.suffix}/std": value
            for key, value in zip(self.topk, stds)
        }
        return {**output_mean, **output_std}


__all__ = ["TopKMetric"]
