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
                calculated per-class and than are averaged over all classes.
                If mode='weighted', metric are calculated per-class and than
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


__all__ = ["IoULoss"]
