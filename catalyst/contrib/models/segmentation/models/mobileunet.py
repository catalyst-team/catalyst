# https://github.com/akirasosa/mobile-semantic-segmentation

import math
import torch
import torch.nn as nn
from torch.nn.functional import interpolate

from ...classification.mobilenetv2 import MobileNetV2, InvertedResidual


class MobileUnet(nn.Module):
    def __init__(
        self, num_classes=1, input_size=224, width_mult=1., pretrained=None
    ):
        super().__init__()

        self.dconv1 = nn.ConvTranspose2d(1280, 96, 4, padding=1, stride=2)
        self.invres1 = InvertedResidual(192, 96, 1, 6)

        self.dconv2 = nn.ConvTranspose2d(96, 32, 4, padding=1, stride=2)
        self.invres2 = InvertedResidual(64, 32, 1, 6)

        self.dconv3 = nn.ConvTranspose2d(32, 24, 4, padding=1, stride=2)
        self.invres3 = InvertedResidual(48, 24, 1, 6)

        self.dconv4 = nn.ConvTranspose2d(24, 16, 4, padding=1, stride=2)
        self.invres4 = InvertedResidual(32, 16, 1, 6)

        self.conv_last = nn.Conv2d(16, 3, 1)
        self.conv_score = nn.Conv2d(3, num_classes, 1)

        self._init_weights()

        self.backbone_encoder = MobileNetV2(
            input_size=input_size,
            width_mult=width_mult,
            pretrained=pretrained
        ).encoder

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x):
        for n in range(0, 2):
            x = self.backbone_encoder[n](x)
        x1 = x

        for n in range(2, 4):
            x = self.backbone_encoder[n](x)
        x2 = x

        for n in range(4, 7):
            x = self.backbone_encoder[n](x)
        x3 = x

        for n in range(7, 14):
            x = self.backbone_encoder[n](x)
        x4 = x

        for n in range(14, 19):
            x = self.backbone_encoder[n](x)
        # x5 = x

        up1 = torch.cat([x4, self.dconv1(x)], dim=1)
        up1 = self.invres1(up1)

        up2 = torch.cat([x3, self.dconv2(up1)], dim=1)
        up2 = self.invres2(up2)

        up3 = torch.cat([x2, self.dconv3(up2)], dim=1)
        up3 = self.invres3(up3)

        up4 = torch.cat([x1, self.dconv4(up3)], dim=1)
        up4 = self.invres4(up4)

        x = up4
        x = interpolate(
            x, scale_factor=2, mode="bilinear", align_corners=False
        )
        x = self.conv_last(x)
        x = self.conv_score(x)

        return x


__all__ = ["MobileUnet"]
