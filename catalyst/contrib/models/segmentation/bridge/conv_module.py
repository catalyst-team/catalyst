import warnings

import torch
import torch.nn as nn

from catalyst.dl.initialization import kaiming_init, constant_init


class ConvModule(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        norm_fn=None,
        activation_fn="relu",
        inplace=True,
        activate_last=True
    ):
        super(ConvModule, self).__init__()
        self.with_norm = norm_fn is not None
        self.with_activation = activation_fn is not None
        self.with_bias = bias
        self.activation = activation_fn
        self.activate_last = activate_last

        if self.with_norm and self.with_bias:
            warnings.warn("ConvModule has norm and bias at the same time")

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias=bias)

        self.in_channels = self.conv.in_channels
        self.out_channels = self.conv.out_channels
        self.kernel_size = self.conv.kernel_size
        self.stride = self.conv.stride
        self.padding = self.conv.padding
        self.dilation = self.conv.dilation
        self.transposed = self.conv.transposed
        self.output_padding = self.conv.output_padding
        self.groups = self.conv.groups

        if self.with_norm:
            norm_channels = out_channels if self.activate_last else in_channels
            self.norm = torch.nn.__dict__[norm_fn](norm_channels)
        else:
            self.norm = None

        if self.with_activation:
            assert activation_fn in ["relu"], "Only ReLU supported."
            if self.activation == "relu":
                self.activate = nn.ReLU(inplace=inplace)

        # Default using msra init
        self._init_weights()

    def _init_weights(self):
        nonlinearity = "relu" if self.activation is None else self.activation
        kaiming_init(self.conv, nonlinearity=nonlinearity)
        if self.with_norm:
            constant_init(self.norm, 1, bias=0)

    def forward(self, x, activate=True, norm=True):
        if self.activate_last:
            x = self.conv(x)
            if norm and self.with_norm:
                x = self.norm(x)
            if activate and self.with_activation:
                x = self.activate(x)
        else:
            if norm and self.with_norm:
                x = self.norm(x)
            if activate and self.with_activation:
                x = self.activate(x)
            x = self.conv(x)
        return x
