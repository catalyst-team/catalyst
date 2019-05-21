from functools import partial
import torch.nn as nn
from catalyst.dl import metrics


class IoULoss(nn.Module):
    """
    Intersection over union (Jaccard) loss
    Args:
        mode (str): one of ``['hard', 'soft']`` to calculate IoU
        eps (float): epsilon to avoid zero division
        threshold (float): threshold for outputs binarization
        activation (str): An torch.nn activation applied to the outputs.
            Must be one of ['none', 'sigmoid', 'softmax2d']
    """
    def __init__(
        self,
        mode: str = "hard",
        eps: float = 1e-7,
        threshold: float = None,
        activation: str = "sigmoid",
    ):
        super().__init__()
        assert mode in ["hard", "soft"], \
            f"Mode must be one of ['hard', 'soft'], got {mode}."
        self.mode = mode

        if mode == "hard":
            self.metric_fn = partial(
                metrics.iou,
                eps=eps,
                threshold=threshold,
                activation=activation)
        else:
            self.metric_fn = partial(
                metrics.soft_iou,
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
        mode (str): one of ``['hard', 'soft']`` to calculate IoU
        eps (float): epsilon to avoid zero division
        threshold (float): threshold for outputs binarization
        activation (str): An torch.nn activation applied to the outputs.
            Must be one of ['none', 'sigmoid', 'softmax2d']
        reduction (str): Specifies the reduction to apply to the output of BCE
    """
    def __init__(
        self,
        mode: str = "hard",
        eps: float = 1e-7,
        threshold: float = None,
        activation: str = "sigmoid",
        reduction: str = "mean",
    ):
        super().__init__()
        self.bce_loss = nn.BCEWithLogitsLoss(reduction=reduction)
        self.iou_loss = IoULoss(mode, eps, threshold, activation)

    def forward(self, outputs, targets):
        iou = self.iou_loss.forward(outputs, targets)
        bce = self.bce_loss(outputs, targets)
        loss = iou + bce
        return loss
