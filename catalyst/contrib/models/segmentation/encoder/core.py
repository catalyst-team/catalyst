from typing import List
from abc import abstractmethod, ABC

import torch
import torch.nn as nn


def _take(elements, indexes):
    return list([elements[i] for i in indexes])


class EncoderSpec(ABC, nn.Module):

    @property
    @abstractmethod
    def layers(self) -> List[int]:
        pass

    @property
    @abstractmethod
    def out_channels(self) -> List[int]:
        pass

    @property
    @abstractmethod
    def out_strides(self) -> List[int]:
        pass

    @property
    @abstractmethod
    def encoder_layers(self) -> List[nn.Module]:
        pass

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        input = x
        output_features = []
        for layer in self.encoder_layers:
            output = layer(input)
            output_features.append(output)
            input = output

        output = _take(output_features, self.layers)
        return output

    def set_requires_grad(self, requires_grad):
        for param in self.parameters():
            param.requires_grad = bool(requires_grad)
