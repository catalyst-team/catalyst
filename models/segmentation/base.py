import numpy as np
from torch import nn
from torch.nn import functional as F
import torch


class Conv2dBnRelu(nn.Module):
    PADDING_METHODS = {'replication': nn.ReplicationPad2d,
                       'reflection': nn.ReflectionPad2d,
                       'zero': nn.ZeroPad2d,
                       }

    def __init__(self, in_channels, out_channels, kernel_size=(3, 3),
                 use_relu=True, use_batch_norm=True, use_padding=True, padding_method='replication'):
        super().__init__()
        self.use_relu = use_relu
        self.use_batch_norm = use_batch_norm
        self.use_padding = use_padding
        self.kernel_w = kernel_size[0]
        self.kernel_h = kernel_size[1]
        self.padding_w = kernel_size[0] - 1
        self.padding_h = kernel_size[1] - 1

        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.padding = Conv2dBnRelu.PADDING_METHODS[padding_method](padding=(0, self.padding_h, self.padding_w, 0))
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=0)

    def forward(self, x):
        if self.use_padding:
            x = self.padding(x)
        x = self.conv(x)
        if self.use_batch_norm:
            x = self.batch_norm(x)
        if self.use_relu:
            x = self.relu(x)
        return x


class DeconvConv2dBnRelu(nn.Module):
    def __init__(self, in_channels, out_channels, use_relu=True, use_batch_norm=True):
        super().__init__()
        self.use_relu = use_relu
        self.use_batch_norm = use_batch_norm

        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3,
                                         stride=2, padding=1, output_padding=1)

    def forward(self, x):
        x = self.deconv(x)
        if self.use_batch_norm:
            x = self.batch_norm(x)
        if self.use_relu:
            x = self.relu(x)
        return x


class NoOperation(nn.Module):
    def forward(self, x):
        return x


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.conv1 = Conv2dBnRelu(in_channels, middle_channels)
        self.conv2 = Conv2dBnRelu(middle_channels, out_channels)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.channel_se = ChannelSELayer(out_channels, reduction=16)
        self.spatial_se = SpatialSELayer(out_channels)

    def forward(self, x, e=None):
        x = self.upsample(x)
        if e is not None:
            x = torch.cat([x, e], 1)
        x = self.conv1(x)
        x = self.conv2(x)

        channel_se = self.channel_se(x)
        spatial_se = self.spatial_se(x)

        x = channel_se + spatial_se
        return x


class ChannelSELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class SpatialSELayer(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.fc = nn.Conv2d(channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.fc(x)
        x = self.sigmoid(x)
        return module_input * x


class DepthChannelExcitation(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.fc = nn.Sequential(nn.Linear(1, channels),
                                nn.Sigmoid()
                                )

    def forward(self, x, d=None):
        b, c, _, _ = x.size()
        y = self.fc(d).view(b, c, 1, 1)
        return x * y


class DepthSpatialExcitation(nn.Module):
    def __init__(self, grid_size=16):
        super().__init__()
        self.grid_size = grid_size
        self.grid_size_sqrt = int(np.sqrt(grid_size))

        self.fc = nn.Sequential(nn.Linear(1, grid_size),
                                nn.Sigmoid()
                                )

    def forward(self, x, d=None):
        b, _, h, w = x.size()
        y = self.fc(d).view(b, 1, self.grid_size_sqrt, self.grid_size_sqrt)
        scale_factor = h // self.grid_size_sqrt
        y = F.upsample(y, scale_factor=scale_factor, mode='bilinear')
        return x * y


class GlobalConvolutionalNetwork(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, use_relu=False):
        super().__init__()

        self.conv1 = nn.Sequential(Conv2dBnRelu(in_channels=in_channels,
                                                out_channels=out_channels,
                                                kernel_size=(kernel_size, 1),
                                                use_relu=use_relu, use_padding=True),
                                   Conv2dBnRelu(in_channels=out_channels,
                                                out_channels=out_channels,
                                                kernel_size=(1, kernel_size),
                                                use_relu=use_relu, use_padding=True),
                                   )
        self.conv2 = nn.Sequential(Conv2dBnRelu(in_channels=in_channels,
                                                out_channels=out_channels,
                                                kernel_size=(1, kernel_size),
                                                use_relu=use_relu, use_padding=True),
                                   Conv2dBnRelu(in_channels=out_channels,
                                                out_channels=out_channels,
                                                kernel_size=(kernel_size, 1),
                                                use_relu=use_relu, use_padding=True),
                                   )

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(x)
        return conv1 + conv2


class BoundaryRefinement(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()

        self.conv = nn.Sequential(Conv2dBnRelu(in_channels=in_channels,
                                               out_channels=out_channels,
                                               kernel_size=(kernel_size, kernel_size),
                                               use_relu=True, use_padding=True),
                                  Conv2dBnRelu(in_channels=in_channels,
                                               out_channels=out_channels,
                                               kernel_size=(kernel_size, kernel_size),
                                               use_relu=False, use_padding=True),
                                  )

    def forward(self, x):
        conv = self.conv(x)
        return x + conv