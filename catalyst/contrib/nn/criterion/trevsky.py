from typing import List, Optional
from functools import partial

import torch
from torch import nn

from catalyst.metrics.functional import trevsky


class TrevskyLoss(nn.Module):
    """The trevsky loss.
    TrevskyIndex = TP / (TP + alpha * FN + betta * FP)
    TrevskyLoss = 1 - TrevskyIndex
    """

    def __init__(
        self,
        alpha: float,
        beta: Optional[float] = None,
        class_dim: int = 1,
        mode: str = "macro",
        weights: List[float] = None,
        eps: float = 1e-7,
    ):
        """
        Args:
            alpha: false negative coefficient, bigger alpha bigger penalty for
                false negative. Must be in (0, 1)
            beta: false positive coefficient, bigger alpha bigger penalty for
                false positive. Must be in (0, 1), if None beta = (1 - alpha)
            class_dim: indicates class dimention (K) for
                ``outputs`` and ``targets`` tensors (default = 1)
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
        self.loss_fn = partial(
            trevsky,
            eps=eps,
            alpha=alpha,
            beta=beta,
            class_dim=class_dim,
            threshold=None,
            mode=mode,
            weights=weights,
        )

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Calculates loss between ``logits`` and ``target`` tensors."""
        trevsky_score = self.loss_fn(outputs, targets)
        return 1 - trevsky_score


class FocalTrevskyLoss(nn.Module):
    """The focal trevsky loss.
    TrevskyIndex = TP / (TP + alpha * FN + betta * FP)
    FocalTrevskyLoss = (1 - TrevskyIndex)^gamma
    Node: focal will use per image, so loss will pay more attention on complicated images
    """

    def __init__(
        self,
        alpha: float,
        beta: Optional[float] = None,
        gamma: float = 4 / 3,
        class_dim: int = 1,
        mode: str = "macro",
        weights: List[float] = None,
        eps: float = 1e-7,
    ):
        """
        Args:
            alpha: false negative coefficient, bigger alpha bigger penalty for
                false negative. Must be in (0, 1)
            beta: false positive coefficient, bigger alpha bigger penalty for
                false positive. Must be in (0, 1), if None beta = (1 - alpha)
            gamma: focal coefficient. It determines how much the weight of
            simple examples is reduced.
            class_dim: indicates class dimention (K) for
                ``outputs`` and ``targets`` tensors (default = 1)
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
        self.gamma = gamma
        self.trevsky_loss = TrevskyLoss(
            alpha=alpha, beta=beta, class_dim=class_dim, mode=mode, weights=weights, eps=eps,
        )

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Calculates loss between ``logits`` and ``target`` tensors."""
        loss = 0
        batch_size = len(outputs)
        for output_sample, target_sample in zip(outputs, targets):
            output_sample = torch.unsqueeze(output_sample, dim=0)
            target_sample = torch.unsqueeze(target_sample, dim=0)
            sample_loss = self.trevsky_loss(output_sample, target_sample)
            loss += sample_loss ** self.gamma
        loss = loss / batch_size  # mean over batch
        return loss


__all__ = ["TrevskyLoss", "FocalTrevskyLoss"]
