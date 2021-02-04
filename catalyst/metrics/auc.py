from typing import Dict

import torch

from catalyst.metrics.functional.auc import auc
from catalyst.metrics.metric import ICallbackLoaderMetric


class AUCMetric(ICallbackLoaderMetric):
    def __init__(self, compute_on_call: bool = True, prefix: str = None, suffix: str = None):
        super().__init__(compute_on_call=compute_on_call, prefix=prefix, suffix=suffix)
        self.metric_name = f"{self.prefix}auc{self.suffix}"
        self.scores = []
        self.targets = []

    def reset(self, num_batches, num_samples) -> None:
        self.scores = []
        self.targets = []

    def update(self, scores: torch.Tensor, targets: torch.Tensor) -> None:
        self.scores.append(scores.cpu().detach())
        self.targets.append(targets.cpu().detach())

    def compute(self) -> torch.Tensor:
        targets = torch.cat(self.targets)
        scores = torch.cat(self.scores)
        score = auc(outputs=scores, targets=targets)
        return score

    def compute_key_value(self) -> Dict[str, float]:
        per_class_auc = self.compute()
        output = {
            f"{self.metric_name}/class_{i:02d}": value.item()
            for i, value in enumerate(per_class_auc)
        }
        output[self.metric_name] = per_class_auc.mean().item()
        return output


__all__ = ["AUCMetric"]
