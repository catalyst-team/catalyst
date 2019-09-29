from pathlib import Path
from typing import Union

import torch
import torch.nn as nn
import torchvision

from catalyst.contrib.modules import Flatten
from catalyst.contrib.registry import MODULES
from catalyst import utils


class ResnetEncoder(nn.Module):
    def __init__(
        self,
        arch: str = "resnet34",
        pretrained: bool = True,
        frozen: bool = True,
        pooling: str = None,
        pooling_kwargs: dict = None,
        cut_layers: int = 2,
        state_dict: Union[dict, str, Path] = None,
    ):
        """
        Specifies an encoder for classification network
        Args:
            arch (str): Name for resnet. Have to be one of
                resnet18, resnet34, resnet50, resnet101, resnet152
            pretrained (bool): If True, returns a model pre-trained on ImageNet
            frozen (bool): If frozen, sets requires_grad to False
            pooling (str): pooling
            pooling_kwargs (dict): params for pooling
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
        if state_dict is not None:
            if isinstance(state_dict, (Path, str)):
                state_dict = torch.load(str(state_dict))
            resnet.load_state_dict(state_dict)

        modules = list(resnet.children())[:-cut_layers]  # delete last layers

        if frozen:
            for module in modules:
                utils.set_requires_grad(module, requires_grad=False)

        if pooling is not None:
            pooling_kwargs = pooling_kwargs or {}
            pooling_layer_fn = MODULES.get(pooling)
            pooling_layer = pooling_layer_fn(
                in_features=resnet.fc.in_features, **pooling_kwargs) \
                if "attn" in pooling.lower() \
                else pooling_layer_fn(**pooling_kwargs)
            modules += [pooling_layer]

            if hasattr(pooling_layer, "out_features"):
                out_features = pooling_layer.out_features(
                    in_features=resnet.fc.in_features
                )
            else:
                out_features = None
        else:
            out_features = resnet.fc.in_features

        modules += [Flatten()]
        self.out_features = out_features

        self.encoder = nn.Sequential(*modules)

    def forward(self, image):
        """Extract the image feature vectors."""
        features = self.encoder(image)
        return features
