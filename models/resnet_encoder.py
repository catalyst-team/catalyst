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
            pooling="GlobalConcatPool2d",
            embedding_size=None, bn_momentum=0.01,
            cut_layers=1,
            **kwargs):
        super().__init__()
        # hack to prevent cycle imports
        from prometheus.modules.modules import name2nn

        resnet = torchvision.models.__dict__[arch](pretrained=pretrained)
        modules = list(resnet.children())[:-cut_layers]  # delete last layers

        if frozen:
            for param in modules:
                param.requires_grad = False

        pooling = name2nn(pooling)
        flatten = name2nn("Flatten")
        modules += [
            pooling(),
            flatten(),
        ]

        if embedding_size is not None:
            resnet_out_features = resnet.fc.in_features \
                if "concat" not in pooling.lower() \
                else resnet.fc.in_features * 2
            additional_modules = [
                nn.Linear(resnet_out_features, embedding_size),
                nn.BatchNorm1d(
                    num_features=embedding_size,
                    momentum=bn_momentum)
            ]
            embeddings_weights_init(additional_modules)
            modules += additional_modules

        self.feature_net = nn.Sequential(*modules)

    def forward(self, image):
        """Extract the image feature vectors."""
        features = self.feature_net(image)
        return features
