from typing import List
from abc import ABC, abstractmethod

from torch import nn


def _take(elements, indexes):
    return [elements[i] for i in indexes]


class EncoderSpec(ABC, nn.Module):
    """@TODO: Docs. Contribution is welcome."""

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


__all__ = ["EncoderSpec"]
