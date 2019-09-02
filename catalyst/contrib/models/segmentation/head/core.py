from typing import List
from abc import abstractmethod, ABC

import torch
import torch.nn as nn


class HeadSpec(ABC, nn.Module):
    def __init__(
        self,
        in_channels: List[int],
        out_channles: int,
        in_strides: List[int] = None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.in_strides = in_strides
        self.out_channles = out_channles

    @abstractmethod
    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        pass
