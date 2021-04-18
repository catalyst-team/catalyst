from typing import Dict, Tuple

import torch

from catalyst.metrics._metric import ICallbackLoaderMetric
from catalyst.metrics.functional._auc import auc, binary_auc
from catalyst.metrics.functional._misc import process_multilabel_components
from catalyst.utils.distributed import all_gather, get_rank


class AUCMetric(ICallbackLoaderMetric):
    """AUC metric,

    Args:
        compute_on_call: if True, computes and returns metric value during metric call
        prefix: metric prefix
        suffix: metric suffix

    .. warning::

        This metric is under API improvement.
    """

    def __init__(self, compute_on_call: bool = True, prefix: str = None, suffix: str = None):
        """Init."""
        super().__init__(compute_on_call=compute_on_call, prefix=prefix, suffix=suffix)
        self.metric_name = f"{self.prefix}auc{self.suffix}"
        self.scores = []
        self.targets = []
        self._is_ddp = get_rank() > -1

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

    def compute(self) -> Tuple[torch.Tensor, float, float, float]:
        """Computes the AUC metric based on saved statistics."""
        targets = torch.cat(self.targets)
        scores = torch.cat(self.scores)

        # @TODO: ddp hotfix, could be done better
        if self._is_ddp:
            scores = torch.cat(all_gather(scores))
            targets = torch.cat(all_gather(targets))

        scores, targets, _ = process_multilabel_components(outputs=scores, targets=targets)
        per_class = auc(scores=scores, targets=targets)
        micro = binary_auc(scores=scores.view(-1), targets=targets.view(-1))[0]
        macro = per_class.mean().item()
        weights = targets.sum(axis=0) / len(targets)
        weighted = (per_class * weights).sum().item()
        return per_class, micro, macro, weighted

    def compute_key_value(self) -> Dict[str, float]:
        """Computes the AUC metric based on saved statistics and returns key-value results."""
        per_class_auc, micro_auc, macro_auc, weighted_auc = self.compute()
        output = {
            f"{self.metric_name}/class_{i:02d}": value.item()
            for i, value in enumerate(per_class_auc)
        }
        output[f"{self.metric_name}/_micro"] = micro_auc
        output[self.metric_name] = macro_auc
        output[f"{self.metric_name}/_macro"] = macro_auc
        output[f"{self.metric_name}/_weighted"] = weighted_auc
        return output


__all__ = ["AUCMetric"]
