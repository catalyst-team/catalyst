import math

import torch.nn as nn

from catalyst.contrib.registry import MODULES
from ..classification import MobileNetV2


class MobileNetV2Encoder(nn.Module):
    def __init__(
        self,
        input_size=224,
        width_mult=1.,
        pretrained=True,
        pooling=None,
        pooling_kwargs=None,
    ):
        super().__init__()

        net = MobileNetV2(
            input_size=input_size,
            width_mult=width_mult,
            pretrained=pretrained)
        self.encoder = list(net.encoder.children())

        if pooling is not None:
            pooling_kwargs = pooling_kwargs or {}
            pooling_layer_fn = MODULES.get(pooling)
            pooling_layer = pooling_layer_fn(
                in_features=self.last_channel, **pooling_kwargs) \
                if "attn" in pooling.lower() \
                else pooling_layer_fn(**pooling_kwargs)
            self.encoder.append(pooling_layer)

            out_features = pooling_layer.out_features(
                in_features=net.output_channel
            )
        else:
            out_features = net.output_channel

        self.out_features = out_features
        # make it nn.Sequential
        self.encoder = nn.Sequential(*self.encoder)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.encoder(x)
        return x
