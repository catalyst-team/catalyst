from typing import Dict

import torch

from catalyst.metrics._metric import ICallbackLoaderMetric
from catalyst.metrics.functional._auc import auc
from catalyst.utils.distributed import all_gather, get_rank


class AUCMetric(ICallbackLoaderMetric):
    """AUC metric,

    Args:
        compute_on_call: if True, computes and returns metric value during metric call
        prefix: metric prefix
        suffix: metric suffix
    """

    def __init__(self, compute_on_call: bool = True, prefix: str = None, suffix: str = None):
        """Init."""
        super().__init__(compute_on_call=compute_on_call, prefix=prefix, suffix=suffix)
        self.metric_name = f"{self.prefix}auc{self.suffix}"
        self.scores = []
        self.targets = []
        self._is_ddp = False

    def reset(self, num_batches, num_samples) -> None:
        """Resets all fields"""
        self._is_ddp = get_rank() > -1
        self.scores = []
        self.targets = []

    def update(self, scores: torch.Tensor, targets: torch.Tensor) -> None:
        """Updates metric value with statistics for new data.

        Args:
            scores: tensor with scores
            targets: tensor with targets
        """
        self.scores.append(scores.cpu().detach())
        self.targets.append(targets.cpu().detach())

    def compute(self) -> torch.Tensor:
        """Computes the AUC metric based on saved statistics."""
        targets = torch.cat(self.targets)
        scores = torch.cat(self.scores)

        # @TODO: ddp hotfix, could be done better
        if self._is_ddp:
            scores = torch.cat(all_gather(scores))
            targets = torch.cat(all_gather(targets))

        score = auc(outputs=scores, targets=targets)
        return score

    def compute_key_value(self) -> Dict[str, float]:
        """Computes the AUC metric based on saved statistics and returns key-value results."""
        per_class_auc = self.compute()
        output = {
            f"{self.metric_name}/class_{i:02d}": value.item()
            for i, value in enumerate(per_class_auc)
        }
        output[self.metric_name] = per_class_auc.mean().item()
        return output


__all__ = ["AUCMetric"]
