import torch
import torch.nn.functional as F


def sigmoid_focal_loss(
    input: torch.Tensor,
    target: torch.Tensor,
    gamma=2.0,
    alpha=0.25,
    reduction="mean"
):
    """
    Compute binary focal loss between target and output logits.

    Source: https://github.com/BloodAxe/pytorch-toolbelt
    See :class:`~pytorch_toolbelt.losses.FocalLoss` for details.

    Args:
        input: Tensor of arbitrary shape
        target: Tensor of the same shape as input
        reduction (string, optional):
            Specifies the reduction to apply to the output:
            "none" | "mean" | "sum" | "batchwise_mean".
            "none": no reduction will be applied,
            "mean": the sum of the output will be divided by the number of
                elements in the output,
            "sum": the output will be summed.

    References::

        https://github.com/open-mmlab/mmdetection/blob/master/mmdet/core/loss/losses.py
    """
    target = target.type(input.type())

    logpt = -F.binary_cross_entropy_with_logits(
        input, target, reduction="none")
    pt = torch.exp(logpt)

    # compute the loss
    loss = -((1 - pt).pow(gamma)) * logpt

    if alpha is not None:
        loss = loss * (alpha * target + (1 - alpha) * (1 - target))

    if reduction == "mean":
        loss = loss.mean()
    if reduction == "sum":
        loss = loss.sum()
    if reduction == "batchwise_mean":
        loss = loss.sum(0)

    return loss
