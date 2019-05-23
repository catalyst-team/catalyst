from typing import List
from collections import OrderedDict

import torch
import torch.nn as nn
import torchvision

from .core import EncoderSpec, _take


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
        requires_grad: bool = False,
        layers: List[int] = None
    ):
        super().__init__()

        resnet = torchvision.models.__dict__[arch](pretrained=pretrained)
        resnet_params = RESNET_PARAMS[arch]
        self._layers = layers or [1, 2, 3, 4]
        self._channels, self._strides = \
            resnet_params["channels"], resnet_params["strides"]

        self.layer0 = nn.Sequential(
            OrderedDict([
                ("conv1", resnet.conv1),
                ("bn1", resnet.bn1),
                ("relu", resnet.relu)
            ])
        )
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        self.set_requires_grad(requires_grad)

    @property
    def layers(self) -> List[int]:
        return self._layers

    @property
    def out_channels(self) -> List[int]:
        return _take(self._channels, self._layers)

    @property
    def out_strides(self) -> List[int]:
        return _take(self._strides, self._layers)

    @property
    def encoder_layers(self):
        return [
            self.layer0,
            self.layer1,
            self.layer2,
            self.layer3,
            self.layer4
        ]

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        input = x
        output_features = []
        for layer in self.encoder_layers:
            output = layer(input)
            output_features.append(output)

            if layer == self.layer0:
                # Fist maxpool operator is not a part of layer0
                # because we want that layer0 output to have stride of 2
                output = self.maxpool(output)
            input = output

        output = _take(output_features, self.layers)
        return output
