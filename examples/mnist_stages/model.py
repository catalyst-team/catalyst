# flake8: noqa
import torch
from torch import nn

from catalyst.contrib.nn import Normalize
from catalyst.contrib.nn.modules import Flatten


class SimpleNet(nn.Module):
    """Simple MNIST convolutional network for test purposes."""

    def __init__(self, num_hidden1=128, num_hidden2=64):
        """
        Args:
            num_hidden1: size of the first hidden representation
            num_hidden2: size of the second hidden representation
        """
        super().__init__()
        self.conv_net = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            Flatten(),
        )
        self.linear_net = nn.Sequential(
            nn.Linear(9216, num_hidden1),
            nn.ReLU(),
            nn.Linear(num_hidden1, num_hidden2),
            Normalize(),
        )
        self._net = nn.Sequential(self.conv_net, self.linear_net)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: input 1d image tensor with the size of [28 x 28]

        Returns:
            extracted features
        """
        return self._net(x)


__all__ = ["SimpleNet"]
