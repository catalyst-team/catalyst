from typing import List
from functools import partial

import torch
from torch import nn

from catalyst.metrics.functional import dice


class DiceLoss(nn.Module):
    """The Dice loss.
    DiceLoss = 1 - dice score
    dice score = 2 * intersection / (intersection + union)) = \
    = 2 * tp / (2 * tp + fp + fn)
    """

    def __init__(
        self,
        class_dim: int = 1,
        mode: str = "macro",
        weights: List[float] = None,
        eps: float = 1e-7,
    ):
        """
        Args:
            class_dim: indicates class dimention (K) for
                ``outputs`` and ``targets`` tensors (default = 1)
            mode: class summation strategy. Must be one of ['micro', 'macro',
                'weighted']. If mode='micro', classes are ignored, and metric
                are calculated generally. If mode='macro', metric are
                calculated per-class and than are averaged over all classes.
                If mode='weighted', metric are calculated per-class and than
                summed over all classes with weights.
            weights: class weights(for mode="weighted")
            eps: epsilon to avoid zero division
        """
        super().__init__()
        assert mode in ["micro", "macro", "weighted"]
        self.loss_fn = partial(
            dice, eps=eps, class_dim=class_dim, threshold=None, mode=mode, weights=weights,
        )

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Calculates loss between ``logits`` and ``target`` tensors."""
        dice_score = self.loss_fn(outputs, targets)
        return 1 - dice_score


__all__ = ["DiceLoss"]
