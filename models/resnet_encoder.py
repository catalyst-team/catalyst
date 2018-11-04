import torch.nn as nn
import torchvision


def embeddings_weights_init(modules):
    for layer in modules:
        if isinstance(layer, nn.Linear):
            layer.weight.data.normal_(0.0, 0.02)
            layer.bias.data.fill_(0)


class ResnetEncoder(nn.Module):
    def __init__(
            self,
            arch="resnet50", pretrained=True, frozen=True,
            pooling=None, pooling_kwargs=None,
            embedding_size=None, bn_momentum=0.01,
            cut_layers=2):
        super().__init__()
        # hack to prevent cycle imports
        from catalyst.modules.modules import name2nn

        resnet = torchvision.models.__dict__[arch](pretrained=pretrained)
        modules = list(resnet.children())[:-cut_layers]  # delete last layers

        if frozen:
            for param in modules:
                param.requires_grad = False

        if pooling is not None:
            pooling_kwargs = pooling_kwargs or {}
            pooling_layer = name2nn(pooling)
            pooling_fn = pooling_layer(
                in_features=resnet.fc.in_features, **pooling_kwargs) \
                if "attn" in pooling.lower() \
                else pooling_layer(**pooling_kwargs)
            modules += [pooling_fn]

            # @TODO: refactor
            if "concatattn" in pooling.lower():
                resnet_out_features = resnet.fc.in_features * 3
            elif any([x in  pooling.lower()
                      for x in ["concat", "avgattn", "maxattn"]]):
                resnet_out_features = resnet.fc.in_features * 2
            else:
                resnet_out_features = resnet.fc.in_features
        else:
            resnet_out_features = resnet.fc.in_features

        flatten = name2nn("Flatten")
        modules += [flatten()]

        if embedding_size is not None:
            additional_modules = [
                nn.Linear(resnet_out_features, embedding_size),
                nn.BatchNorm1d(
                    num_features=embedding_size,
                    momentum=bn_momentum)
            ]
            embeddings_weights_init(additional_modules)
            modules += additional_modules
            self.out_features = embedding_size
        else:
            self.out_features = resnet_out_features

        self.feature_net = nn.Sequential(*modules)

    def forward(self, image):
        """Extract the image feature vectors."""
        features = self.feature_net(image)
        return features
