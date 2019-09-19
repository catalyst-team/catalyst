from typing import List, Union
from collections import OrderedDict

from pathlib import Path

import torch
import torch.nn as nn
import torchvision

from .core import EncoderSpec, _take
from catalyst import utils

RESNET_PARAMS = {
    "resnet18": {
        "channels": [64, 64, 128, 256, 512],
        "strides": [2, 4, 8, 16, 32]
    },
    "resnet34": {
        "channels": [64, 64, 128, 256, 512],
        "strides": [2, 4, 8, 16, 32]
    },
    "resnet50": {
        "channels": [64, 256, 512, 1024, 2048],
        "strides": [2, 4, 8, 16, 32]
    },
    "resnet101": {
        "channels": [64, 256, 512, 1024, 2048],
        "strides": [2, 4, 8, 16, 32]
    },
    "resnet152": {
        "channels": [64, 256, 512, 1024, 2048],
        "strides": [2, 4, 8, 16, 32]
    },
}


class ResnetEncoder(EncoderSpec):
    def __init__(
        self,
        arch: str = "resnet18",
        pretrained: bool = True,
        requires_grad: bool = True,
        layers_indices: List[int] = None,
        state_dict: Union[dict, str, Path] = None,
    ):
        """
        Specifies an encoder for segmentation network
        Args:
            arch (str): Name for resnet. Have to be one of
                resnet18, resnet34, resnet50, resnet101, resnet152
            pretrained (bool): If True, returns a model pre-trained on ImageNet
            requires_grad (bool): Flag for set_requires_grad.
                If None, calculates as ``not requires_grad``
            layers_indices (List[int]): layers of encoder used for segmentation
                If None, calculates as ``[1, 2, 3, 4]``
            state_dict (Union[dict, str, Path]): Path to ``torch.Model``
                or a dict containing parameters and persistent buffers.
        Examples:
            >>> encoder = ResnetEncoder(
            >>>    arch="resnet18",
            >>>    pretrained=False,
            >>>    state_dict="/model/path/resnet18-5c106cde.pth"
            >>> )
        """
        super().__init__()

        resnet = torchvision.models.__dict__[arch](pretrained=pretrained)
        resnet_params = RESNET_PARAMS[arch]
        if state_dict is not None:
            if isinstance(state_dict, (Path, str)):
                state_dict = torch.load(str(state_dict))
            resnet.load_state_dict(state_dict)
        self._layers_indices = layers_indices or [1, 2, 3, 4]
        self._channels, self._strides = \
            resnet_params["channels"], resnet_params["strides"]
        self._channels = _take(self._channels, self._layers_indices)
        self._strides = _take(self._strides, self._layers_indices)

        layer0 = nn.Sequential(
            OrderedDict(
                [
                    ("conv1", resnet.conv1), ("bn1", resnet.bn1),
                    ("relu", resnet.relu)
                ]
            )
        )
        self._layers = nn.ModuleList(
            [
                layer0, resnet.layer1, resnet.layer2, resnet.layer3,
                resnet.layer4
            ]
        )
        self.maxpool0 = resnet.maxpool

        if requires_grad is None:
            requires_grad = not pretrained

        utils.set_requires_grad(self, requires_grad)

    @property
    def out_channels(self) -> List[int]:
        return self._channels

    @property
    def out_strides(self) -> List[int]:
        return self._strides

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        output = []
        for i, layer in enumerate(self._layers):
            layer_output = layer(x)
            output.append(layer_output)

            if i == 0:
                # Fist maxpool operator is not a part of layer0
                # because we want that layer0 output to have stride of 2
                layer_output = self.maxpool0(layer_output)
            x = layer_output

        output = _take(output, self._layers_indices)
        return output
