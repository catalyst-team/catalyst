import torch
from torch import nn

from catalyst.contrib.nn import Normalize
from catalyst.contrib.nn.modules import Flatten


class MnistSimpleNet(nn.Module):
    """Simple MNIST convolutional network for test purposes."""

    def __init__(self, out_features: int, normalize: bool = True):
        """
        Args:
            out_features: size of the output tensor
        """
        super().__init__()
        layers = [
            nn.Conv2d(1, 32, 3, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            Flatten(),
            nn.Linear(9216, 128),
            nn.ReLU(),
            nn.Linear(128, out_features),
        ]
        if normalize:
            layers.append(Normalize())
        self._net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: input 1d image tensor with the size of [28 x 28]

        Returns:
            extracted features
        """
        return self._net(x)


__all__ = ["MnistSimpleNet"]
