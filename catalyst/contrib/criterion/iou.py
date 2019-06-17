from functools import partial
import torch.nn as nn
from catalyst.dl.utils import criterion


class IoULoss(nn.Module):
    """
    Intersection over union (Jaccard) loss

    Args:
        eps (float): epsilon to avoid zero division
        threshold (float): threshold for outputs binarization
        activation (str): An torch.nn activation applied to the outputs.
            Must be one of ['none', 'Sigmoid', 'Softmax2d']
    """
    def __init__(
        self,
        eps: float = 1e-7,
        threshold: float = None,
        activation: str = "Sigmoid",
    ):
        super().__init__()
        self.metric_fn = partial(
            criterion.iou,
            eps=eps,
            threshold=threshold,
            activation=activation)

    def forward(self, outputs, targets):
        iou = self.metric_fn(outputs, targets)
        return 1 - iou


class BCEIoULoss(nn.Module):
    """
    Intersection over union (Jaccard) with BCE loss

    Args:
        eps (float): epsilon to avoid zero division
        threshold (float): threshold for outputs binarization
        activation (str): An torch.nn activation applied to the outputs.
            Must be one of ['none', 'Sigmoid', 'Softmax2d']
        reduction (str): Specifies the reduction to apply to the output of BCE
    """
    def __init__(
        self,
        eps: float = 1e-7,
        threshold: float = None,
        activation: str = "Sigmoid",
        reduction: str = "mean",
    ):
        super().__init__()
        self.bce_loss = nn.BCEWithLogitsLoss(reduction=reduction)
        self.iou_loss = IoULoss(eps, threshold, activation)

    def forward(self, outputs, targets):
        iou = self.iou_loss.forward(outputs, targets)
        bce = self.bce_loss(outputs, targets)
        loss = iou + bce
        return loss
