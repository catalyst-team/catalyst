import torch.nn as nn
import torchvision


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
        # hack to prevent cycle imports
        from catalyst.contrib.registry import Registry

        resnet = torchvision.models.__dict__[arch](pretrained=pretrained)
        modules = list(resnet.children())[:-cut_layers]  # delete last layers

        if frozen:
            for module in modules:
                for param in module.parameters():
                    param.requires_grad = False

        if pooling is not None:
            pooling_kwargs = pooling_kwargs or {}
            pooling_layer = Registry.name2nn(pooling)
            pooling_fn = pooling_layer(
                in_features=resnet.fc.in_features, **pooling_kwargs) \
                if "attn" in pooling.lower() \
                else pooling_layer(**pooling_kwargs)
            modules += [pooling_fn]

            resnet_out_features = pooling_fn.out_features(
                in_features=resnet.fc.in_features
            )
        else:
            resnet_out_features = resnet.fc.in_features

        flatten = Registry.name2nn("Flatten")
        modules += [flatten()]
        self.out_features = resnet_out_features

        self.feature_net = nn.Sequential(*modules)

    def forward(self, image):
        """Extract the image feature vectors."""
        features = self.feature_net(image)
        return features
