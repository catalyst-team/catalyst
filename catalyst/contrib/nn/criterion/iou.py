# flake8: noqa
# @TODO: code formatting issue for 20.07 release
from functools import partial

from torch import nn

from catalyst.utils import metrics


class IoULoss(nn.Module):
    """The intersection over union (Jaccard) loss.

    @TODO: Docs. Contribution is welcome.
    """

    def __init__(
        self,
        eps: float = 1e-7,
        threshold: float = None,
        activation: str = "Sigmoid",
    ):
        """
        Args:
            eps: epsilon to avoid zero division
            threshold: threshold for outputs binarization
            activation: An torch.nn activation applied to the outputs.
                Must be one of ``'none'``, ``'Sigmoid'``, ``'Softmax2d'``
        """
        super().__init__()
        self.metric_fn = partial(
            metrics.iou, eps=eps, threshold=threshold, activation=activation
        )

    def forward(self, outputs, targets):
        """@TODO: Docs. Contribution is welcome."""
        iou = self.metric_fn(outputs, targets)
        return 1 - iou


class BCEIoULoss(nn.Module):
    """The Intersection over union (Jaccard) with BCE loss.

    @TODO: Docs. Contribution is welcome.
    """

    def __init__(
        self,
        eps: float = 1e-7,
        threshold: float = None,
        activation: str = "Sigmoid",
        reduction: str = "mean",
    ):
        """
        Args:
            eps: epsilon to avoid zero division
            threshold: threshold for outputs binarization
            activation: An torch.nn activation applied to the outputs.
                Must be one of ``'none'``, ``'Sigmoid'``, ``'Softmax2d'``
            reduction: Specifies the reduction to apply
                to the output of BCE
        """
        super().__init__()
        self.bce_loss = nn.BCEWithLogitsLoss(reduction=reduction)
        self.iou_loss = IoULoss(eps, threshold, activation)

    def forward(self, outputs, targets):
        """@TODO: Docs. Contribution is welcome."""
        iou = self.iou_loss.forward(outputs, targets)
        bce = self.bce_loss(outputs, targets)
        loss = iou + bce
        return loss


__all__ = ["IoULoss", "BCEIoULoss"]
