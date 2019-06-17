"""
Creates a MobileNetV2 Model as defined in:
Mark Sandler, Andrew Howard, Menglong Zhu,
Andrey Zhmoginov, Liang-Chieh Chen. (2018).
MobileNetV2: Inverted Residuals and Linear Bottlenecks
arXiv preprint arXiv:1801.04381.
import from https://github.com/tonylins/pytorch-mobilenet-v2

source: https://github.com/d-li14/mobilenetv2.pytorch
"""

import math
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

model_urls = {
    "1.0": "https://github.com/catalyst-team/model_zoo/raw/master/mobilenet_v2/mobilenetv2-91156ee6c9cbc9d320b9414ddaf52a1681f35f5088ab74b08722b292cb34d665.pth",  # noqa:E501
    "0.5": "https://github.com/catalyst-team/model_zoo/raw/master/mobilenet_v2/mobilenetv2_0.5-b23b05d4da1172bc16cd3e9536198f60c53dc20a230301c13c2980df1ab8c613.pth",  # noqa:E501
    "0.25": "https://github.com/catalyst-team/model_zoo/raw/master/mobilenet_v2/mobilenetv2_0.25-dd945b24220c9296162de0362e22f56f2b3f82c072d98e22548eb371194b1549.pth"  # noqa:E501
}


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def conv_3x3_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super().__init__()
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.identity = stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(
                    hidden_dim, hidden_dim, 3, stride, 1,
                    groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(
                    hidden_dim, hidden_dim, 3, stride, 1,
                    groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(
        self,
        num_classes=1000,
        input_size=224,
        width_mult=1.0,
        pretrained=None
    ):
        super().__init__()

        # building first layer
        assert input_size % 32 == 0
        input_channel = _make_divisible(32 * width_mult, 8)
        layers = [conv_3x3_bn(3, input_channel, 2)]

        # setting of inverted residual blocks
        inverted_residual_settings = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]
        # building inverted residual blocks
        block = InvertedResidual
        for t, c, n, s in inverted_residual_settings:
            output_channel = _make_divisible(c * width_mult, 8)
            layers.append(block(input_channel, output_channel, s, t))
            input_channel = output_channel
            for i in range(1, n):
                layers.append(block(input_channel, output_channel, 1, t))
                input_channel = output_channel

        # building last several layers
        self.output_channel = _make_divisible(1280 * width_mult, 8) \
            if width_mult > 1.0 \
            else 1280
        layers.append(conv_1x1_bn(input_channel, self.output_channel))

        self.layers = layers
        self.encoder = nn.Sequential(*layers)
        self.avgpool = nn.AvgPool2d(input_size // 32, stride=1)
        self.classifier = nn.Linear(self.output_channel, num_classes)

        self._initialize_weights()

        if pretrained is not None:
            pretrained = str(width_mult)
            if pretrained in model_urls:
                state_dict = model_zoo.load_url(model_urls[pretrained])
                self.load_state_dict(state_dict)
            else:
                raise NotImplementedError

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
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


__all__ = ["MobileNetV2"]
