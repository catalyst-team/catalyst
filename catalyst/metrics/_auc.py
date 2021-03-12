from typing import Dict

import torch

from catalyst.metrics._metric import ICallbackLoaderMetric
from catalyst.metrics.functional._auc import auc
from catalyst.utils.distributed import all_gather, get_rank


class AUCMetric(ICallbackLoaderMetric):
    """@TODO: docs here

    Args:
        compute_on_call: @TODO: docs
        prefix: @TODO: docs
        suffix:@TODO: docs
    """

    def __init__(self, compute_on_call: bool = True, prefix: str = None, suffix: str = None):
        """Init."""
        super().__init__(compute_on_call=compute_on_call, prefix=prefix, suffix=suffix)
        self.metric_name = f"{self.prefix}auc{self.suffix}"
        self.scores = []
        self.targets = []
        self._is_ddp = False

    def reset(self, num_batches, num_samples) -> None:
        """@TODO: docs here"""
        self._is_ddp = get_rank() > -1
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

        # @TODO: ddp hotfix, could be done better
        if self._is_ddp:
            scores = torch.cat(all_gather(scores))
            targets = torch.cat(all_gather(targets))

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
