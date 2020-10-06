# flake8: noqa
# @TODO: code formatting issue for 20.07 release
import torch
from torch import nn
from torch.nn import functional as F


class Flatten(nn.Module):
    """Flattens the input. Does not affect the batch size.

    @TODO: Docs (add `Example`). Contribution is welcome.
    """

    def __init__(self):
        """@TODO: Docs. Contribution is welcome."""
        super().__init__()

    def forward(self, x):
        """Forward call."""
        return x.view(x.shape[0], -1)


class Lambda(nn.Module):
    """@TODO: Docs. Contribution is welcome."""

    def __init__(self, lambda_fn):
        """@TODO: Docs. Contribution is welcome."""
        super().__init__()
        self.lambda_fn = lambda_fn

    def forward(self, x):
        """Forward call."""
        return self.lambda_fn(x)


class Normalize(nn.Module):
    """Performs :math:`L_p` normalization of inputs over specified dimension.

    @TODO: Docs (add `Example`). Contribution is welcome.
    """

    def __init__(self, **normalize_kwargs):
        """
        Args:
            **normalize_kwargs: see ``torch.nn.functional.normalize`` params
        """
        super().__init__()
        self.normalize_kwargs = normalize_kwargs

    def forward(self, x):
        """Forward call."""
        return F.normalize(x, **self.normalize_kwargs)


class GaussianNoise(nn.Module):
    """
    A gaussian noise module.

    Shape:

    - Input: (batch, \*)
    - Output: (batch, \*) (same shape as input)
    """

    def __init__(self, stddev: float = 0.1):
        """
        Args:
            stddev: The standard deviation of the normal distribution.
                Default: 0.1.
        """
        super().__init__()
        self.stddev = stddev

    def forward(self, x: torch.Tensor):
        """Forward call."""
        noise = torch.empty_like(x)
        noise.normal_(0, self.stddev)


__all__ = ["Flatten", "Lambda", "Normalize", "GaussianNoise"]
