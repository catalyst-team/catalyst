# flake8: noqa
# @TODO: code formatting issue for 20.07 release
import math

import torch
from torch import nn
from torch.nn import functional as F

from catalyst.registry import REGISTRY


class GlobalAvgPool2d(nn.Module):
    """Applies a 2D global average pooling operation over an input signal
    composed of several input planes.

    @TODO: Docs (add `Example`). Contribution is welcome.
    """

    def __init__(self):
        """Constructor method for the ``GlobalAvgPool2d`` class."""
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward call."""
        h, w = x.shape[2:]
        return F.avg_pool2d(input=x, kernel_size=(h, w))

    @staticmethod
    def out_features(in_features):
        """Returns number of channels produced by the pooling.

        Args:
            in_features: number of channels in the input sample

        Returns:
            number of output features
        """
        return in_features


class GlobalMaxPool2d(nn.Module):
    """Applies a 2D global max pooling operation over an input signal
    composed of several input planes.

    @TODO: Docs (add `Example`). Contribution is welcome.
    """

    def __init__(self):
        """Constructor method for the ``GlobalMaxPool2d`` class."""
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward call."""
        h, w = x.shape[2:]
        return F.max_pool2d(input=x, kernel_size=(h, w))

    @staticmethod
    def out_features(in_features):
        """Returns number of channels produced by the pooling.

        Args:
            in_features: number of channels in the input sample

        Returns:
            number of output features
        """
        return in_features


class GlobalConcatPool2d(nn.Module):
    """@TODO: Docs (add `Example`). Contribution is welcome."""

    def __init__(self):
        """Constructor method for the ``GlobalConcatPool2d`` class."""
        super().__init__()
        self.avg = GlobalAvgPool2d()
        self.max = GlobalMaxPool2d()  # noqa: WPS125

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward call."""
        return torch.cat([self.avg(x), self.max(x)], 1)

    @staticmethod
    def out_features(in_features):
        """Returns number of channels produced by the pooling.

        Args:
            in_features: number of channels in the input sample

        Returns:
            number of output features
        """
        return in_features * 2


class GlobalAttnPool2d(nn.Module):
    """@TODO: Docs. Contribution is welcome."""

    def __init__(self, in_features, activation_fn="Sigmoid"):
        """@TODO: Docs. Contribution is welcome."""
        super().__init__()

        activation_fn = REGISTRY.get_if_str(activation_fn)
        self.attn = nn.Sequential(
            nn.Conv2d(in_features, 1, kernel_size=1, stride=1, padding=0, bias=False),
            activation_fn(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward call."""
        x_a = self.attn(x)
        x = x * x_a
        x = torch.sum(x, dim=[-2, -1], keepdim=True)
        return x

    @staticmethod
    def out_features(in_features):
        """Returns number of channels produced by the pooling.

        Args:
            in_features: number of channels in the input sample

        Returns:
            number of output features
        """
        return in_features


class GlobalAvgAttnPool2d(nn.Module):
    """@TODO: Docs (add `Example`). Contribution is welcome."""

    def __init__(self, in_features, activation_fn="Sigmoid"):
        """@TODO: Docs. Contribution is welcome."""
        super().__init__()
        self.avg = GlobalAvgPool2d()
        self.attn = GlobalAttnPool2d(in_features, activation_fn)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward call."""
        return torch.cat([self.avg(x), self.attn(x)], 1)

    @staticmethod
    def out_features(in_features):
        """Returns number of channels produced by the pooling.

        Args:
            in_features: number of channels in the input sample

        Returns:
            number of output features
        """
        return in_features * 2


class GlobalMaxAttnPool2d(nn.Module):
    """@TODO: Docs (add `Example`). Contribution is welcome."""

    def __init__(self, in_features, activation_fn="Sigmoid"):
        """@TODO: Docs. Contribution is welcome."""
        super().__init__()
        self.max = GlobalMaxPool2d()  # noqa: WPS125
        self.attn = GlobalAttnPool2d(in_features, activation_fn)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward call."""
        return torch.cat([self.max(x), self.attn(x)], 1)

    @staticmethod
    def out_features(in_features):
        """Returns number of channels produced by the pooling.

        Args:
            in_features: number of channels in the input sample

        Returns:
            number of output features
        """
        return in_features * 2


class GlobalConcatAttnPool2d(nn.Module):
    """@TODO: Docs (add `Example`). Contribution is welcome."""

    def __init__(self, in_features, activation_fn="Sigmoid"):
        """@TODO: Docs. Contribution is welcome."""
        super().__init__()
        self.avg = GlobalAvgPool2d()
        self.max = GlobalMaxPool2d()  # noqa: WPS125
        self.attn = GlobalAttnPool2d(in_features, activation_fn)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward call."""
        return torch.cat([self.avg(x), self.max(x), self.attn(x)], 1)

    @staticmethod
    def out_features(in_features):
        """Returns number of channels produced by the pooling.

        Args:
            in_features: number of channels in the input sample

        Returns:
            number of output features
        """
        return in_features * 3


class GeM2d(nn.Module):
    """Implementation of GeM: Generalized Mean Pooling.
    Example:
        >>> x = torch.randn(2,1280,8,8) #output of last convolutional layer of the network
        >>> gem_pool = GeM2d(p = 2.2 , p_trainable = False)
        >>> op = gem_pool(x)
        >>> op.shape
        torch.Size([1, 1280, 1, 1])
        >>> op
        tensor([[[[1.0660]],
             [[1.1599]],
             [[0.5934]],
             ...,
             [[0.6889]],
             [[1.0361]],
             [[0.9717]]]], grad_fn=<PowBackward0>)
    """

    def __init__(self, p: float = 3.0, p_trainable: bool = False, eps: float = 1e-7):
        """
        Args:
            p: The pooling parameter.
                Default: 3.0
            p_trainable: Whether the pooling parameter(p) should be trainable.
                    Default: False
            eps: epsilon for numerical stability.
        """
        super().__init__()
        if p_trainable:
            # if p_trainable is True and the value of p is set to math.inf

            if p not in [math.inf, float("inf")]:

                self.p = nn.Parameter(torch.ones(1) * p)
            else:

                self.p = math.inf

        else:
            self.p = p

        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward call."""
        h, w = x.shape[2:]

        if self.p in [math.inf, float("inf")]:
            # if p-> inf return max pooled features
            return F.max_pool2d(x, kernel_size=(h, w))

        else:
            x = x.clamp(min=self.eps).pow(self.p)
            return F.avg_pool2d(x, kernel_size=(h, w)).pow(1.0 / self.p)

    @staticmethod
    def out_features(in_features):
        """Returns number of channels produced by the pooling.

        Args:
            in_features: number of channels in the input sample.

        Returns:
            number of output features
        """
        return in_features


__all__ = [
    "GlobalAttnPool2d",
    "GlobalAvgAttnPool2d",
    "GlobalAvgPool2d",
    "GlobalConcatAttnPool2d",
    "GlobalConcatPool2d",
    "GlobalMaxAttnPool2d",
    "GlobalMaxPool2d",
    "GeM2d",
]
