from typing import Any, Dict, List

import torch

from catalyst.metrics.additive import AdditiveValueMetric
from catalyst.metrics.functional.accuracy import accuracy
from catalyst.metrics.functional.misc import get_default_topk_args
from catalyst.metrics.metric import ICallbackBatchMetric


class AccuracyMetric(ICallbackBatchMetric):
    def __init__(
        self,
        num_classes: int = None,
        topk_args: List[int] = None,
        compute_on_call: bool = True,
        prefix: str = None,
        suffix: str = None,
    ):
        super().__init__(compute_on_call=compute_on_call, prefix=prefix, suffix=suffix)
        self.metric_name_mean = f"{self.prefix}accuracy{self.suffix}"
        self.metric_name_std = f"{self.prefix}accuracy{self.suffix}/std"
        self.topk_args: List[int] = topk_args or get_default_topk_args(num_classes)
        self.additive_metrics: List[AdditiveValueMetric] = [
            AdditiveValueMetric() for _ in range(len(self.topk_args))
        ]

    def reset(self) -> None:
        for metric in self.additive_metrics:
            metric.reset()

    def update(self, logits: torch.Tensor, targets: torch.Tensor) -> List[float]:
        values = accuracy(logits, targets, topk=self.topk_args)
        values = [v.item() for v in values]
        for value, metric in zip(values, self.additive_metrics):
            metric.update(value, len(targets))
        return values

    def update_key_value(self, logits: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        values = self.update(logits=logits, targets=targets)
        output = {
            f"{self.prefix}accuracy{key:02d}{self.suffix}": value
            for key, value in zip(self.topk_args, values)
        }
        output[self.metric_name_mean] = output[f"{self.prefix}accuracy01{self.suffix}"]
        return output

    def compute(self) -> Any:
        means, stds = zip(*(metric.compute() for metric in self.additive_metrics))
        return means, stds

    def compute_key_value(self) -> Dict[str, float]:
        means, stds = self.compute()
        output_mean = {
            f"{self.prefix}accuracy{key:02d}{self.suffix}": value
            for key, value in zip(self.topk_args, means)
        }
        output_std = {
            f"{self.prefix}accuracy{key:02d}{self.suffix}/std": value
            for key, value in zip(self.topk_args, stds)
        }
        output_mean[self.metric_name_mean] = output_mean[f"{self.prefix}accuracy01{self.suffix}"]
        output_std[self.metric_name_std] = output_std[f"{self.prefix}accuracy01{self.suffix}/std"]
        return {**output_mean, **output_std}


__all__ = ["AccuracyMetric"]
