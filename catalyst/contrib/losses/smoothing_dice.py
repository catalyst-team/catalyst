from typing import Callable, List
from functools import partial

import torch
from torch import nn


class SmoothingDiceLoss(nn.Module):
    """
    The Smoothing Dice loss.
    ``SmoothingDiceloss = 1 - smoothing dice score``
    ``smoothing dice score = 2 * intersection / (|outputs|^2 + |targets|^2)``
    Criterion was inspired by https://arxiv.org/abs/1606.04797

    Examples:
        >>> import torch
        >>> from catalyst.contrib.losses import SmoothingDiceLoss
        >>> targets = torch.abs(torch.randn((1, 2, 3, 3), ))
        >>> prediction = torch.abs(torch.randn((1, 2, 3, 3)))
        >>> criterion = SmoothingDiceLoss()
        >>> loss = criterion(prediction, targets)
        >>> print(loss)
    """

    def __init__(
        self,
        class_dim: int = 1,
        mode: str = "macro",
        weights: List[float] = None,
        eps: float = 1e-7,
    ):
        """
        Args:
            class_dim: indicates class dimention (K) for
                ``outputs`` and ``targets`` tensors (default = 1)
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
        self.class_dim = class_dim
        self.mode = mode
        if self.mode == "weighted":
            assert weights is not None
            self.weights = torch.Tensor(weights)
        self.eps = eps

    def _get_sum_per_class(
        self, outputs_shape: List[int]
    ) -> Callable[[torch.Tensor], torch.Tensor]:
        """
        Creates a channel summing function

        Args:
            outputs_shape: shape of output tensor

        Returns:
             function that sums tensors over all channels except the
             classification

        """
        n_dims = len(outputs_shape)
        dims = list(range(n_dims))
        # support negative index
        if self.class_dim < 0:
            self.class_dim = n_dims + self.class_dim
        dims.pop(self.class_dim)
        sum_per_class = partial(torch.sum, dim=dims)
        return sum_per_class

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Calculates loss between ``logits`` and ``target`` tensors."""
        sum_per_class = self._get_sum_per_class(outputs.shape)
        outputs_2 = outputs ** 2
        targets_2 = targets ** 2
        tp = sum_per_class(outputs * targets)
        outputs_per_class = sum_per_class(outputs_2)
        targets_per_class = sum_per_class(targets_2)
        if self.mode == "micro":
            tp = tp.sum()
            outputs_per_class = outputs_per_class.sum()
            targets_per_class = targets_per_class.sum()
        smoothing_dice = 2 * tp / (outputs_per_class + targets_per_class + self.eps)
        if self.mode == "macro":
            smoothing_dice = smoothing_dice.mean()
        if self.mode == "weighted":
            device = smoothing_dice.device
            smoothing_dice = (smoothing_dice * self.weights.to(device)).sum()
        return 1 - smoothing_dice
