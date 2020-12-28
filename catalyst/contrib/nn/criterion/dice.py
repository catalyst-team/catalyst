# flake8: noqa
# @TODO: code formatting issue for 20.07 release
from typing import List
from functools import partial

import torch
from torch import nn

from catalyst.metrics.functional import wrap_metric_fn_with_activation
from catalyst.metrics.region_base_metrics import dice


class DiceLoss(nn.Module):
    """The Dice loss.
    DiceLoss = 1 - dice score
    dice score = 2 * intersection / (intersection + union)) = \
    = 2 * tp / (2 * tp + fp + fn)
    """

    def __init__(
        self,
        class_dim: int = 1,
        activation: str = "Sigmoid",
        mode: str = "micro",
        weights: List[float] = None,
        eps: float = 1e-7,
    ):
        """
        Args:
            class_dim: indicates class dimention (K) for
                ``outputs`` and ``targets`` tensors (default = 1)
            activation: An torch.nn activation applied to the outputs.
                Must be one of ``'none'``, ``'Sigmoid'``, ``'Softmax'``
            mode: class summation strategy. Must be one of ['micro', 'macro',
                'weighted']. If mode='micro', classes are ignored, and metric
                are calculated generally. If mode='macro', metric are
                calculated separately and than are averaged over all classes.
                If mode='weighted', metric are calculated separately and than
                summed over all classes with weights.
            weights: class weights(for mode="weighted")
            eps: epsilon to avoid zero division
        """
        super().__init__()
        assert mode in ["micro", "macro", "weighted"]
        metric_fn = wrap_metric_fn_with_activation(
            metric_fn=dice, activation=activation
        )
        self.loss_fn = partial(
            metric_fn,
            eps=eps,
            class_dim=class_dim,
            threshold=None,
            mode=mode,
            weights=weights,
        )

    def forward(
        self, outputs: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Calculates loss between ``logits`` and ``target`` tensors."""
        dice_score = self.loss_fn(outputs, targets)
        return 1 - dice_score


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
                "Both bce_wight and dice_weight cannot be "
                "equal to 0 at the same time."
            )

        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

        if self.bce_weight != 0:
            self.bce_loss = nn.BCEWithLogitsLoss()

        if self.dice_weight != 0:
            self.dice_loss = DiceLoss(eps=eps, activation=activation)

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
