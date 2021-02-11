rom typing import Any, Dict, List

import torch

from catalyst.metrics.additive import AdditiveValueMetric
from catalyst.metrics.functional.hitrate import hitrate
from catalyst.metrics.functional.misc import get_default_topk_args
from catalyst.metrics.metric import ICallbackBatchMetric


class HitrateMetric(ICallbackBatchMetric):
    def __init__(
        self,
        topk_args: List[int] = [1],
        compute_on_call: bool = True,
        prefix: str = None,
        suffix: str = None,
    ):
        super().__init__(compute_on_call=compute_on_call, prefix=prefix, suffix=suffix)
        self.metric_name = f"{self.prefix}hitrate{self.suffix}"
        self.topk_args: List[int] = topk_args
        self.additive_metrics: List[AdditiveValueMetric] = [
            AdditiveValueMetric() for _ in range(len(self.topk_args))
        ]

    def reset(self) -> None:
        for metric in self.additive_metrics:
            metric.reset()

    def update(self, logits: torch.Tensor, targets: torch.Tensor) -> List[float]:
        values = hitrate(logits, targets, topk=self.topk_args)
        values = [v.item() for v in values]
        for value, metric in zip(values, self.additive_metrics):
            metric.update(value, len(targets))
        return values

    def update_key_value(self, logits: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        values = self.update(logits=logits, targets=targets)
        output = {
            f"{self.prefix}hitrate{key:02d}{self.suffix}": value
            for key, value in zip(self.topk_args, values)
        }
        output[self.metric_name_mean] = output[f"{self.prefix}hitrate01{self.suffix}"]
        return output

    def compute(self) -> Any:
        means, stds = zip(*(metric.compute() for metric in self.additive_metrics))
        return means, stds

    def compute_key_value(self) -> Dict[str, float]:
        means, stds = self.compute()
        output = {
            f"{self.prefix}hitrate{key:02d}{self.suffix}": value
            for key, value in zip(self.topk_args, means)
        }
        output_mean[self.metric_name] = output[f"{self.prefix}hitrate{self.suffix}"]
        return {**output_mean, **output_std}


__all__ = ["HitrateMetric"]
