from typing import Dict

import torch

from catalyst.metrics._metric import ICallbackLoaderMetric
from catalyst.metrics.functional._auc import auc


class AUCMetric(ICallbackLoaderMetric):
    """@TODO: docs here"""
    def __init__(self, compute_on_call: bool = True, prefix: str = None, suffix: str = None):
        """@TODO: docs here"""
        super().__init__(compute_on_call=compute_on_call, prefix=prefix, suffix=suffix)
        self.metric_name = f"{self.prefix}auc{self.suffix}"
        self.scores = []
        self.targets = []

    def reset(self, num_batches, num_samples) -> None:
        """@TODO: docs here"""
        self.scores = []
        self.targets = []

    def update(self, scores: torch.Tensor, targets: torch.Tensor) -> None:
        """@TODO: docs here"""
        self.scores.append(scores.cpu().detach())
        self.targets.append(targets.cpu().detach())

    def compute(self) -> torch.Tensor:
        """@TODO: docs here"""
        targets = torch.cat(self.targets)
        scores = torch.cat(self.scores)
        score = auc(outputs=scores, targets=targets)
        return score

    def compute_key_value(self) -> Dict[str, float]:
        """@TODO: docs here"""
        per_class_auc = self.compute()
        output = {
            f"{self.metric_name}/class_{i:02d}": value.item()
            for i, value in enumerate(per_class_auc)
        }
        output[self.metric_name] = per_class_auc.mean().item()
        return output


__all__ = ["AUCMetric"]
