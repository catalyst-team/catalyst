# flake8: noqa
# @TODO: code formatting issue for 20.07 release
from functools import partial

import torch
from torch import nn

from catalyst.metrics.dice import dice
from catalyst.metrics.functional import wrap_metric_fn_with_activation


class DiceLoss(nn.Module):
    """@TODO: Docs. Contribution is welcome."""

    def __init__(
        self, eps: float = 1e-7, threshold: float = None, activation: str = "Sigmoid",
    ):
        """@TODO: Docs. Contribution is welcome."""
        super().__init__()

        metric_fn = wrap_metric_fn_with_activation(metric_fn=dice, activation=activation)
        self.loss_fn = partial(metric_fn, eps=eps, threshold=threshold)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        """Calculates loss between ``logits`` and ``target`` tensors.

        Args:
            logits: model logits
            targets: ground truth labels

        Returns:
            computed loss
        """
        per_class_dice = self.loss_fn(logits, targets)  # [bs; num_classes]
        dice = torch.mean(per_class_dice)
        return 1 - dice


class BCEDiceLoss(nn.Module):
    """@TODO: Docs. Contribution is welcome."""

    def __init__(
        self,
        eps: float = 1e-7,
        threshold: float = None,
        activation: str = "Sigmoid",
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
    ):
        """@TODO: Docs. Contribution is welcome."""
        super().__init__()

        if bce_weight == 0 and dice_weight == 0:
            raise ValueError(
                "Both bce_wight and dice_weight cannot be " "equal to 0 at the same time."
            )

        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

        if self.bce_weight != 0:
            self.bce_loss = nn.BCEWithLogitsLoss()

        if self.dice_weight != 0:
            self.dice_loss = DiceLoss(eps=eps, threshold=threshold, activation=activation)

    def forward(self, outputs, targets):
        """@TODO: Docs. Contribution is welcome."""
        if self.bce_weight == 0:
            return self.dice_weight * self.dice_loss(outputs, targets)
        if self.dice_weight == 0:
            return self.bce_weight * self.bce_loss(outputs, targets)

        dice = self.dice_weight * self.dice_loss(outputs, targets)
        bce = self.bce_weight * self.bce_loss(outputs, targets)
        return dice + bce


__all__ = ["BCEDiceLoss", "DiceLoss"]
