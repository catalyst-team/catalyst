# flake8: noqa
# TODO: add docs and refactor
from functools import partial
import math

import torch
from torch import nn


def wing_loss(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    width: int = 5,
    curvature: float = 0.5,
    reduction: str = "mean",
) -> torch.Tensor:
    """The Wing loss.

    It has been proposed in `Wing Loss for Robust Facial Landmark Localisation
    with Convolutional Neural Networks`_.

    Args:
        @TODO: Docs. Contribution is welcome.

    Adapted from:
    https://github.com/BloodAxe/pytorch-toolbelt (MIT License)

    .. _Wing Loss for Robust Facial Landmark Localisation with Convolutional
        Neural Networks: https://arxiv.org/abs/1711.06753
    """
    diff_abs = (targets - outputs).abs()
    loss = diff_abs.clone()

    idx_smaller = diff_abs < width
    idx_bigger = diff_abs >= width

    loss[idx_smaller] = width * torch.log(1 + diff_abs[idx_smaller] / curvature)

    c = width - width * math.log(1 + width / curvature)
    loss[idx_bigger] = loss[idx_bigger] - c

    if reduction == "sum":
        loss = loss.sum()
    if reduction == "mean":
        loss = loss.mean()

    return loss


class WingLoss(nn.Module):
    """Creates a criterion that optimizes a Wing loss.

    It has been proposed in `Wing Loss for Robust Facial Landmark Localisation
    with Convolutional Neural Networks`_.

    Adapted from:
    https://github.com/BloodAxe/pytorch-toolbelt

    .. _Wing Loss for Robust Facial Landmark Localisation with Convolutional
        Neural Networks: https://arxiv.org/abs/1711.06753
    """

    def __init__(self, width: int = 5, curvature: float = 0.5, reduction: str = "mean"):
        """
        Args:
            @TODO: Docs. Contribution is welcome.
        """
        super().__init__()
        self.loss_fn = partial(wing_loss, width=width, curvature=curvature, reduction=reduction)

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            @TODO: Docs. Contribution is welcome.
        """
        loss = self.loss_fn(outputs, targets)
        return loss


__all__ = ["WingLoss"]
