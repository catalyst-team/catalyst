import torch
from torch import nn
from torch.nn.functional import relu, max_pool2d

from catalyst.contrib.nn import Normalize


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
        self.conv1 = nn.Conv2d(input_channels, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, features_dim)
        self.norm = Normalize()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: input image tensor

        Returns:
            extracted features
        """
        x = relu(self.conv1(x))
        x = relu(self.conv2(x))
        x = max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = relu(self.fc1(x))
        x = self.fc2(x)
        x = self.norm(x)
        return x