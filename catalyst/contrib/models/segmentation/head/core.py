from typing import List
from abc import abstractmethod, ABC

import torch
import torch.nn as nn


class HeadSpec(ABC, nn.Module):

    @abstractmethod
    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        pass

    def set_requires_grad(self, requires_grad):
        for param in self.parameters():
            param.requires_grad = bool(requires_grad)
