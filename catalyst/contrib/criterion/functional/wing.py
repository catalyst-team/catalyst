import math
import torch


def wing_loss(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    width: int = 5,
    curvature: float = 0.5,
    reduction: str = "mean"
):
    """
    https://arxiv.org/pdf/1711.06753.pdf

    Source https://github.com/BloodAxe/pytorch-toolbelt
    See :class:`~pytorch_toolbelt.losses` for details.
    """
    diff_abs = (targets - outputs).abs()
    loss = diff_abs.clone()

    idx_smaller = diff_abs < width
    idx_bigger = diff_abs >= width

    loss[idx_smaller] = \
        width * torch.log(1 + diff_abs[idx_smaller] / curvature)

    C = width - width * math.log(1 + width / curvature)
    loss[idx_bigger] = loss[idx_bigger] - C

    if reduction == "sum":
        loss = loss.sum()
    if reduction == "mean":
        loss = loss.mean()

    return loss
