from typing import Any, Dict, List

import torch

from catalyst.metrics._additive import AdditiveValueMetric
from catalyst.metrics._metric import ICallbackBatchMetric
from catalyst.metrics.functional._misc import get_default_topk_args
from catalyst.metrics.functional._ndcg import ndcg


class NDCGMetric(ICallbackBatchMetric):
    def __init__(
        self,
        topk_args: List[int] = [1],
        compute_on_call: bool = True,
        prefix: str = None,
        suffix: str = None,
    ):
        super().__init__(compute_on_call=compute_on_call, prefix=prefix, suffix=suffix)
        self.metric_name_mean = f"{self.prefix}ndcg{self.suffix}"
        self.metric_name_std = f"{self.prefix}ndcg{self.suffix}/std"
        self.topk_args: List[int] = topk_args
        self.additive_metrics: List[AdditiveValueMetric] = [
            AdditiveValueMetric() for _ in range(len(self.topk_args))
        ]

    def reset(self) -> None:
        for metric in self.additive_metrics:
            metric.reset()

    def update(self, logits: torch.Tensor, targets: torch.Tensor) -> List[float]:
        values = ndcg(logits, targets, topk=self.topk_args)
        values = [v.item() for v in values]
        for value, metric in zip(values, self.additive_metrics):
            metric.update(value, len(targets))
        return values

    def update_key_value(self, logits: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        values = self.update(logits=logits, targets=targets)
        output = {
            f"{self.prefix}ndcg{key:02d}{self.suffix}": value
            for key, value in zip(self.topk_args, values)
        }
        output[self.metric_name_mean] = output[f"{self.prefix}ndcg01{self.suffix}"]
        return output

    def compute(self) -> Any:
        means, stds = zip(*(metric.compute() for metric in self.additive_metrics))
        return means, stds

    def compute_key_value(self) -> Dict[str, float]:
        means, stds = self.compute()
        output_mean = {
            f"{self.prefix}ndcg{key:02d}{self.suffix}": value
            for key, value in zip(self.topk_args, means)
        }
        output_std = {
            f"{self.prefix}ndcg{key:02d}{self.suffix}/std": value
            for key, value in zip(self.topk_args, stds)
        }
        output_mean[self.metric_name_mean] = output_mean[f"{self.prefix}ndcg01{self.suffix}"]
        output_std[self.metric_name_std] = output_std[f"{self.prefix}ndcg01{self.suffix}/std"]
        return {**output_mean, **output_std}


__all__ = ["NDCGMetric"]
