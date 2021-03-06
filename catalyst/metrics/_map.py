from typing import Any, Dict, List

import torch

from catalyst.metrics._additive import AdditiveValueMetric
from catalyst.metrics._metric import ICallbackBatchMetric
from catalyst.metrics.functional._average_precision import (
    mean_average_precision,
)


class MAPMetric(ICallbackBatchMetric):
    """@TODO: docs here"""

    def __init__(
        self,
        topk_args: List[int] = None,
        compute_on_call: bool = True,
        prefix: str = None,
        suffix: str = None,
    ):
        """@TODO: docs here"""
        super().__init__(
            compute_on_call=compute_on_call, prefix=prefix, suffix=suffix
        )
        self.metric_name_mean = f"{self.prefix}map{self.suffix}"
        self.metric_name_std = f"{self.prefix}map{self.suffix}/std"
        self.topk_args: List[int] = topk_args or [1]
        self.additive_metrics: List[AdditiveValueMetric] = [
            AdditiveValueMetric() for _ in range(len(self.topk_args))
        ]

    def reset(self) -> None:
        """@TODO: docs here"""
        for metric in self.additive_metrics:
            metric.reset()

    def update(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> List[float]:
        """@TODO: docs here"""
        values = mean_average_precision(logits, targets, topk=self.topk_args)
        values = [v.item() for v in values]
        for value, metric in zip(values, self.additive_metrics):
            metric.update(value, len(targets))
        return values

    def update_key_value(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> Dict[str, float]:
        """@TODO: docs here"""
        values = self.update(logits=logits, targets=targets)
        output = {
            f"{self.prefix}map{key:02d}{self.suffix}": value
            for key, value in zip(self.topk_args, values)
        }
        output[self.metric_name_mean] = output[
            f"{self.prefix}map01{self.suffix}"
        ]
        return output

    def compute(self) -> Any:
        """@TODO: docs here"""
        means, stds = zip(
            *(metric.compute() for metric in self.additive_metrics)
        )
        return means, stds

    def compute_key_value(self) -> Dict[str, float]:
        """@TODO: docs here"""
        means, stds = self.compute()
        output_mean = {
            f"{self.prefix}map{key:02d}{self.suffix}": value
            for key, value in zip(self.topk_args, means)
        }
        output_std = {
            f"{self.prefix}map{key:02d}{self.suffix}/std": value
            for key, value in zip(self.topk_args, stds)
        }
        output_mean[self.metric_name_mean] = output_mean[
            f"{self.prefix}map01{self.suffix}"
        ]
        output_std[self.metric_name_std] = output_std[
            f"{self.prefix}map01{self.suffix}/std"
        ]
        return {**output_mean, **output_std}


__all__ = ["MAPMetric"]
