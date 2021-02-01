# flake8: noqa
# @TODO: code formatting issue for 20.07 release
from functools import partial

import torch
from torch import nn

from catalyst.metrics.functional.dice import dice


class DiceLoss(nn.Module):
    """@TODO: Docs. Contribution is welcome."""

    def __init__(self, eps: float = 1e-7, threshold: float = None):
        """@TODO: Docs. Contribution is welcome."""
        super().__init__()

        self.loss_fn = partial(dice, eps=eps, threshold=threshold)

    def forward(self, scores: torch.Tensor, targets: torch.Tensor):
        """Calculates loss between ``logits`` and ``target`` tensors.

        Args:
            scores: model logits
            targets: ground truth labels

        Returns:
            computed loss
        """
        per_class_dice = self.loss_fn(scores, targets)  # [bs; num_classes]
        dice = torch.mean(per_class_dice)
        return 1 - dice


__all__ = ["DiceLoss"]
