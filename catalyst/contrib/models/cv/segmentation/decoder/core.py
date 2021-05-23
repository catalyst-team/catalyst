# flake8: noqa
# @TODO: code formatting issue for 20.07 release
from typing import List
from abc import ABC, abstractmethod

import torch
from torch import nn


class DecoderSpec(ABC, nn.Module):
    """@TODO: Docs. Contribution is welcome."""

    def __init__(self, in_channels: List[int], in_strides: List[int]):
        """
        Args:
            in_channels: number of channels in the input sample
            in_strides: the stride of the block
        """
        super().__init__()
        self.in_channels = in_channels
        self.in_strides = in_strides

    @property
    @abstractmethod
    def out_channels(self) -> List[int]:
        """Number of channels produced by the block."""
        pass

    @property
    @abstractmethod
    def out_strides(self) -> List[int]:
        """@TODO: Docs. Contribution is welcome."""
        pass

    @abstractmethod
    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        """Forward call."""
        pass


__all__ = ["DecoderSpec"]
