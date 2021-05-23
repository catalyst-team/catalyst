# flake8: noqa
# @TODO: code formatting issue for 20.07 release
import torch
from torch import nn


class RMSNorm(nn.Module):
    """An implementation of RMS Normalization.

    @TODO: Docs (link to paper). Contribution is welcome.
    """

    def __init__(self, dimension: int, epsilon: float = 1e-8, is_bias: bool = False):
        """
        Args:
            dimension: the dimension of the layer output to normalize
            epsilon: an epsilon to prevent dividing by zero
                in case the layer has zero variance. (default = 1e-8)
            is_bias: a boolean value whether to include bias term
                while normalization
        """
        super().__init__()
        self.dimension = dimension
        self.epsilon = epsilon
        self.is_bias = is_bias
        self.scale = nn.Parameter(torch.ones(self.dimension))
        if self.is_bias:
            self.bias = nn.Parameter(torch.zeros(self.dimension))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """@TODO: Docs. Contribution is welcome."""
        x_std = torch.sqrt(torch.mean(x ** 2, -1, keepdim=True))
        x_norm = x / (x_std + self.epsilon)
        if self.is_bias:
            return self.scale * x_norm + self.bias
        return self.scale * x_norm


__all__ = ["RMSNorm"]
