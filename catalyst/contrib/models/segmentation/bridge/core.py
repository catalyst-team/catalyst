from typing import List
from abc import abstractmethod, ABC

import torch
import torch.nn as nn


class BridgeSpec(ABC, nn.Module):
    def __init__(self, in_channels: List[int], in_strides: List[int]):
        super().__init__()
        self._in_channels = in_channels
        self._in_strides = in_strides

    @property
    def in_channels(self) -> List[int]:
        return self._in_channels

    @property
    def in_strides(self) -> List[int]:
        return self._in_strides

    @property
    @abstractmethod
    def out_channels(self) -> List[int]:
        pass

    @property
    @abstractmethod
    def out_strides(self) -> List[int]:
        pass

    @abstractmethod
    def forward(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        pass
