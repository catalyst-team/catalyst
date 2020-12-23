# flake8: noqa
# @TODO: code formatting issue for 20.07 release
from functools import partial
from typing import Optional, List

import torch
from torch import nn

from catalyst.metrics.functional import wrap_metric_fn_with_activation
from catalyst.metrics.functional import get_aggregated_metric
from catalyst.metrics.region_base_metrics import trevsky


class TrevskyLoss(nn.Module):
    """The trevsky loss."""

    def __init__(
        self,
        alpha: float,
        class_dim: int = 1,
        activation: str = "Sigmoid",
        beta: Optional[float] = None,
        mode: str = 'mean',
        weights: List[float] = None,
        eps: float = 1e-7,
    ):
        """
        Args:
            alpha: false negative coefficient, bigger alpha bigger penalty for
                false negative. Must be in (0, 1)
            class_dim: indicates class dimention (K) for
                ``outputs`` and ``targets`` tensors (default = 1)
            activation: An torch.nn activation applied to the outputs.
                Must be one of ``'none'``, ``'Sigmoid'``, ``'Softmax'``
            beta: false positive coefficient, bigger alpha bigger penalty for
                false positive. Must be in (0, 1), if None beta = (1 - alpha)
            mode: class summation strategy. Must be one of
                ["mean", "sum", "weighted"]
            weights: class weights(for mode="weighted")
            eps: epsilon to avoid zero division
        """
        super().__init__()
        self.mode = mode
        self.weights = weights
        metric_fn = wrap_metric_fn_with_activation(
            metric_fn=trevsky, activation=activation
        )
        self.loss_fn = partial(metric_fn,
                               eps=eps,
                               alpha=alpha,
                               beta=beta,
                               class_dim=class_dim,
                               threshold=None)

    def forward(self,
                outputs: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        """Calculates loss between ``logits`` and ``target`` tensors."""
        per_class_trevsky = self.loss_fn(outputs, targets)  # [bs; num_classes]
        score = get_aggregated_metric(per_class_trevsky,
                                      mode=self.mode,
                                      weights=self.weights)
        return 1 - score


__all__ = ["TrevskyLoss"]
