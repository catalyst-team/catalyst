from abc import abstractmethod, ABC

import torch
import torch.nn as nn


class EncoderBlock(ABC, nn.Module):

    @property
    @abstractmethod
    def block(self) -> nn.Module:
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CentralBlock(ABC, nn.Module):

    @property
    @abstractmethod
    def block(self) -> nn.Module:
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DecoderBlock(ABC, nn.Module):

    def __init__(
        self,
        in_channels: int,
        enc_channels: int,
        out_channels: int,
        **kwargs
    ):
        super().__init__()
        self.in_channels = in_channels
        self.enc_channels = enc_channels
        self.out_channels = out_channels
        pass

    @property
    @abstractmethod
    def block(self) -> nn.Module:
        pass

    @abstractmethod
    def forward(self, down: torch.Tensor, left: torch.Tensor) -> torch.Tensor:
        pass
