# flake8: noqa
# @TODO: code formatting issue for 20.07 release
from typing import List
from functools import partial

import torch
from torch import nn

from catalyst.metrics.functional import wrap_metric_fn_with_activation
from catalyst.metrics.region_base_metrics import iou


class IoULoss(nn.Module):
    """The intersection over union (Jaccard) loss.
    IOULoss = 1 - iou score
    iou score = intersection / union = tp / (tp + fp + fn)
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
            metric_fn=iou, activation=activation
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
        iou_score = self.loss_fn(outputs, targets)
        return 1 - iou_score


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
                Must be one of ``'none'``, ``'Sigmoid'``, ``'Softmax'``
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
