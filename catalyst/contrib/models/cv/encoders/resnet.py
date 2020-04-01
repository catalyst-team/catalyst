from typing import Union
from pathlib import Path

import torch
from torch import nn
import torchvision

from catalyst import utils
from catalyst.contrib.nn.modules import Flatten
from catalyst.contrib.registry import MODULES


class ResnetEncoder(nn.Module):
    """Specifies ResNet encoders for classification network.

    Examples:
        >>> encoders = ResnetEncoder(
        >>>    arch="resnet18",
        >>>    pretrained=False,
        >>>    state_dict="/model/path/resnet18-5c106cde.pth"
        >>> )
    """

    def __init__(
        self,
        arch: str = "resnet18",
        pretrained: bool = True,
        frozen: bool = True,
        pooling: str = None,
        pooling_kwargs: dict = None,
        cut_layers: int = 2,
        state_dict: Union[dict, str, Path] = None,
    ):
        """
        Args:
            arch (str): Name for resnet. Have to be one of
                resnet18, resnet34, resnet50, resnet101, resnet152
            pretrained (bool): If True, returns a model pre-trained on ImageNet
            frozen (bool): If frozen, sets requires_grad to False
            pooling (str): pooling
            pooling_kwargs (dict): params for pooling
            state_dict (Union[dict, str, Path]): Path to ``torch.Model``
                or a dict containing parameters and persistent buffers.
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
            pooling_layer = (
                pooling_layer_fn(
                    in_features=resnet.fc.in_features, **pooling_kwargs
                )
                if "attn" in pooling.lower()
                else pooling_layer_fn(**pooling_kwargs)
            )
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
