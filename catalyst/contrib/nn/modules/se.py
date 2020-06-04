import torch
import torch.nn as nn
import torch.nn.functional as F


class SqueezeAndExcitation(nn.Module):
    """
    The channel-wise SE (Squeeze and Excitation) block from the
    [Squeeze-and-Excitation Networks](https://arxiv.org/abs/1709.01507) paper.

    Adapted from
    https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/65939
    and
    https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/66178

    Shape:

    - Input: (batch, channels, height, width)
    - Output: (batch, channels, height, width) (same shape as input)
    """

    def __init__(self, in_channels: int, r: int = 16):
        """
        Args:
            in_channels (int): The number of channels
                in the feature map of the input.
            r (int): The reduction ratio of the intermediate channels.
                Default: 16.
        """
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, in_channels // r)
        self.linear_2 = nn.Linear(in_channels // r, in_channels)

    def forward(self, x: torch.Tensor):
        """Forward call."""
        input_x = x

        x = x.view(*(x.shape[:-2]), -1).mean(-1)
        x = F.relu(self.linear_1(x), inplace=True)
        x = self.linear_2(x)
        x = x.unsqueeze(-1).unsqueeze(-1)
        x = torch.sigmoid(x)

        x = torch.mul(input_x, x)
        return x


class ChannelSqueezeAndSpatialExcitation(nn.Module):
    """
    The sSE (Channel Squeeze and Spatial Excitation) block from the
    [Concurrent Spatial and Channel ‘Squeeze & Excitation’
    in Fully Convolutional Networks](https://arxiv.org/abs/1803.02579) paper.

    Adapted from
    https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/66178

    Shape:

    - Input: (batch, channels, height, width)
    - Output: (batch, channels, height, width) (same shape as input)
    """

    def __init__(self, in_channels: int):
        """
        Args:
            in_channels (int): The number of channels
                in the feature map of the input.
        """
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1, stride=1)

    def forward(self, x: torch.Tensor):
        """Forward call."""
        input_x = x

        x = self.conv(x)
        x = torch.sigmoid(x)

        x = torch.mul(input_x, x)
        return x


class ConcurrentSpatialAndChannelSqueezeAndChannelExcitation(nn.Module):
    """
    The scSE (Concurrent Spatial and Channel Squeeze and Channel Excitation)
    block from the [Concurrent Spatial and Channel ‘Squeeze & Excitation’
    in Fully Convolutional Networks](https://arxiv.org/abs/1803.02579) paper.

    Adapted from
    https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/66178

    Shape:

    - Input: (batch, channels, height, width)
    - Output: (batch, channels, height, width) (same shape as input)
    """

    def __init__(self, in_channels: int, r: int = 16):
        """
        Args:
            in_channels (int): The number of channels
                in the feature map of the input.
            r (int): The reduction ratio of the intermediate channels.
                Default: 16.
        """
        super().__init__()
        self.cse_block = SqueezeAndExcitation(in_channels, r)
        self.sse_block = ChannelSqueezeAndSpatialExcitation(in_channels)

    def forward(self, x: torch.Tensor):
        """Forward call."""
        cse = self.cse_block(x)
        sse = self.sse_block(x)
        x = torch.add(cse, sse)
        return x
