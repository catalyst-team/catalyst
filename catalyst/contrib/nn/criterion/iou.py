# flake8: noqa
# @TODO: code formatting issue for 20.07 release
from functools import partial

import torch
from torch import nn

from catalyst.metrics.iou import iou


class IoULoss(nn.Module):
    """The intersection over union (Jaccard) loss.

    @TODO: Docs. Contribution is welcome.
    """

    def __init__(
        self, eps: float = 1e-7, threshold: float = None,
    ):
        """
        Args:
            eps: epsilon to avoid zero division
            threshold: threshold for outputs binarization
        """
        super().__init__()
        self.loss_fn = partial(iou, eps=eps, threshold=threshold)

    def forward(self, scores, targets):
        """@TODO: Docs. Contribution is welcome."""
        per_class_iou = self.loss_fn(scores, targets)  # [bs; num_classes]
        iou = torch.mean(per_class_iou)
        return 1 - iou


__all__ = ["IoULoss"]
