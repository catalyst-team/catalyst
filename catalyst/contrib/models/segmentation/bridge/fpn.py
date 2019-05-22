from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from catalyst.dl.initialization import xavier_init
from .core import BridgeSpec
from .conv_module import ConvModule


class FPN(BridgeSpec):

    def __init__(
        self,
        in_channels: List[int],
        out_channels: int,
        num_outputs: int = None,
        activation_fn=None,
        norm_fn=None,
        use_extra_convs=False,
    ):
        super().__init__()

        self._in_channels = in_channels
        self._out_channels = out_channels
        self._num_inputs = len(in_channels)
        self._num_outputs = num_outputs or len(in_channels)

        self._activation_fn = activation_fn
        self._norm_fn = norm_fn
        self._use_bias = norm_fn is None
        self._use_extra_convs = use_extra_convs

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(self._num_inputs):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                norm_fn=self._norm_fn,
                bias=self._use_bias,
                activation_fn=self._activation_fn,
                inplace=False)
            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                norm_fn=self._norm_fn,
                bias=self._use_bias,
                activation_fn=self._activation_fn,
                inplace=False)

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        # add extra conv layers (e.g., RetinaNet)
        extra_levels = self._num_outputs - self._num_inputs
        if self._use_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                in_channels = (
                    self.in_channels[-1]
                    if i == 0 else out_channels)
                extra_fpn_conv = ConvModule(
                    in_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                    norm_fn=self._norm_fn,
                    bias=self._use_bias,
                    activation_fn=self._activation_fn,
                    inplace=False)
                self.fpn_convs.append(extra_fpn_conv)

        self._init_weights()

    # default init_weights for conv(msra) and norm in ConvModule
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution="uniform")

    @property
    def in_channels(self) -> List[int]:
        return self._in_channels

    @property
    def out_channels(self) -> List[int]:
        return [self._out_channels] * self._num_outputs

    @property
    def with_extra_convs(self) -> bool:
        return self._use_extra_convs

    def forward(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        inputs = x

        # build laterals
        laterals = [
            lateral_conv(inputs[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            laterals[i - 1] += F.interpolate(
                laterals[i], scale_factor=2, mode="nearest")

        # build outputs
        # part 1: from original levels
        outputs = [
            self.fpn_convs[i](laterals[i]) for i in
            range(used_backbone_levels)
        ]

        # part 2: add extra levels
        if self._num_outputs > len(outputs):
            # add conv layers on top of original feature maps (RetinaNet)
            if self._use_extra_convs:
                original = inputs[-1]
                outputs.append(self.fpn_convs[used_backbone_levels](original))
                for i in range(used_backbone_levels + 1, self._num_outputs):
                    # @TODO: bug
                    # BUG: we should add relu before each extra conv
                    outputs.append(self.fpn_convs[i](outputs[-1]))
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            else:
                for i in range(self._num_outputs - used_backbone_levels):
                    outputs.append(F.max_pool2d(outputs[-1], 1, stride=2))

        return outputs
