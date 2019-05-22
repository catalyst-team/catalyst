from typing import List
from abc import abstractmethod, ABC

import torch
import torch.nn as nn


class BridgeSpec(ABC, nn.Module):

    @property
    @abstractmethod
    def in_channels(self) -> List[int]:
        pass

    @property
    @abstractmethod
    def out_channels(self) -> List[int]:
        pass

    @abstractmethod
    def forward(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        pass

    def set_requires_grad(self, requires_grad):
        for m in self.modules():
            for param in m.parameters():
                param.requires_grad = bool(requires_grad)
