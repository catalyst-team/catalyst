import torch.nn as nn
import torchvision

from catalyst.contrib.modules import Flatten
from catalyst.contrib.registry import MODULES


class ResnetEncoder(nn.Module):
    def __init__(
        self,
        arch="resnet34",
        pretrained=True,
        frozen=True,
        pooling=None,
        pooling_kwargs=None,
        cut_layers=2
    ):
        super().__init__()

        resnet = torchvision.models.__dict__[arch](pretrained=pretrained)
        modules = list(resnet.children())[:-cut_layers]  # delete last layers

        if frozen:
            for module in modules:
                for param in module.parameters():
                    param.requires_grad = False

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
