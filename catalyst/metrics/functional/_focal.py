import torch
import torch.nn.functional as F


def sigmoid_focal_loss(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    gamma: float = 2.0,
    alpha: float = 0.25,
    reduction: str = "mean",
):
    """
    Compute binary focal loss between target and output logits.

    Args:
        outputs: tensor of arbitrary shape
        targets: tensor of the same shape as input
        gamma: gamma for focal loss
        alpha: alpha for focal loss
        reduction (string, optional):
            specifies the reduction to apply to the output:
            ``"none"`` | ``"mean"`` | ``"sum"`` | ``"batchwise_mean"``.
            ``"none"``: no reduction will be applied,
            ``"mean"``: the sum of the output will be divided by the number of
            elements in the output,
            ``"sum"``: the output will be summed.

    Returns:
        computed loss

    Source: https://github.com/BloodAxe/pytorch-toolbelt
    """
    targets = targets.type(outputs.type())

    logpt = -F.binary_cross_entropy_with_logits(outputs, targets, reduction="none")
    pt = torch.exp(logpt)

    # compute the loss
    loss = -((1 - pt).pow(gamma)) * logpt

    if alpha is not None:
        loss = loss * (alpha * targets + (1 - alpha) * (1 - targets))

    if reduction == "mean":
        loss = loss.mean()
    if reduction == "sum":
        loss = loss.sum()
    if reduction == "batchwise_mean":
        loss = loss.sum(0)

    return loss


def reduced_focal_loss(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    gamma: float = 2.0,
    reduction="mean",
) -> torch.Tensor:
    """Compute reduced focal loss between target and output logits.

    It has been proposed in `Reduced Focal Loss\: 1st Place Solution to xView
    object detection in Satellite Imagery`_ paper.

    .. note::
        ``size_average`` and ``reduce`` params are in the process of being
        deprecated, and in the meantime, specifying either of those two args
        will override ``reduction``.

    Source: https://github.com/BloodAxe/pytorch-toolbelt

    .. _Reduced Focal Loss\: 1st Place Solution to xView object detection
        in Satellite Imagery: https://arxiv.org/abs/1903.01347

    Args:
        outputs: tensor of arbitrary shape
        targets: tensor of the same shape as input
        threshold: threshold for focal reduction
        gamma: gamma for focal reduction
        reduction: specifies the reduction to apply to the output:
            ``"none"`` | ``"mean"`` | ``"sum"`` | ``"batchwise_mean"``.
            ``"none"``: no reduction will be applied,
            ``"mean"``: the sum of the output will be divided by the number of
            elements in the output,
            ``"sum"``: the output will be summed.
            ``"batchwise_mean"`` computes mean loss per sample in batch.
            Default: "mean"

    Returns:  # noqa: DAR201
        torch.Tensor: computed loss
    """
    targets = targets.type(outputs.type())

    logpt = -F.binary_cross_entropy_with_logits(outputs, targets, reduction="none")
    pt = torch.exp(logpt)

    # compute the loss
    focal_reduction = ((1.0 - pt) / threshold).pow(gamma)
    focal_reduction[pt < threshold] = 1

    loss = -focal_reduction * logpt

    if reduction == "mean":
        loss = loss.mean()
    if reduction == "sum":
        loss = loss.sum()
    if reduction == "batchwise_mean":
        loss = loss.sum(0)

    return loss


__all__ = ["sigmoid_focal_loss", "reduced_focal_loss"]
