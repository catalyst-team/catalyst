import torch
from torch import nn

from catalyst.contrib.nn import Normalize
from catalyst.contrib.nn.modules import Flatten


class SimpleConv(nn.Module):
    """
    Simple convolutional network.
    """

    def __init__(self, input_channels: int, features_dim: int):
        """
        Args:
            input_channels: number of channels in the 1st layer
            features_dim: size of the output tensor
        """
        super(SimpleConv, self).__init__()
        self._net = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            Flatten(),
            nn.Linear(9216, 128),
            nn.ReLU(),
            nn.Linear(128, features_dim),
            Normalize(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: input image tensor

        Returns:
            extracted features
        """
        return self._net(x)
