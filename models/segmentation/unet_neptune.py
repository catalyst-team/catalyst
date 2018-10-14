from torch import nn
from torch.nn import functional as F
import torch

from .base import Conv2dBnRelu, DecoderBlock
from .encoders import ResNetEncoders, SeResNetEncoders, SeResNetXtEncoders, DenseNetEncoders

"""
This script has been taken (and modified) from :
https://github.com/ternaus/TernausNet
@ARTICLE{arXiv:1801.05746,
         author = {V. Iglovikov and A. Shvets},
          title = {TernausNet: U-Net with VGG11 Encoder Pre-Trained on ImageNet for Image Segmentation},
        journal = {ArXiv e-prints},
         eprint = {1801.05746}, 
           year = 2018
        }
"""


class UNetResNet(nn.Module):
    """PyTorch U-Net model using ResNet(34, 101 or 152) encoder.
    UNet: https://arxiv.org/abs/1505.04597
    ResNet: https://arxiv.org/abs/1512.03385
    Proposed by Alexander Buslaev: https://www.linkedin.com/in/al-buslaev/
    Args:
            encoder_depth (int): Depth of a ResNet encoder (34, 101 or 152).
            num_classes (int): Number of output classes.
            num_filters (int, optional): Number of filters in the last layer of decoder. Defaults to 32.
            dropout_2d (float, optional): Probability factor of dropout layer before output layer. Defaults to 0.2.
            pretrained (bool, optional):
                False - no pre-trained weights are being used.
                True  - ResNet encoder is pre-trained on ImageNet.
                Defaults to False.
            is_deconv (bool, optional):
                False: bilinear interpolation is used in decoder.
                True: deconvolution is used in decoder.
                Defaults to False.
    """

    def __init__(self, encoder_depth, num_classes, dropout_2d=0.0, pretrained=False, use_hypercolumn=False, pool0=False):
        super().__init__()
        self.num_classes = num_classes
        self.dropout_2d = dropout_2d
        self.use_hypercolumn = use_hypercolumn

        self.encoders = ResNetEncoders(encoder_depth, pretrained=pretrained, pool0=pool0)

        if encoder_depth in [18, 34]:
            bottom_channel_nr = 512
        elif encoder_depth in [50, 101, 152]:
            bottom_channel_nr = 2048
        else:
            raise NotImplementedError('only 18, 34, 50, 101, 152 version of Resnet are implemented')

        self.center = nn.Sequential(Conv2dBnRelu(bottom_channel_nr, bottom_channel_nr),
                                    Conv2dBnRelu(bottom_channel_nr, bottom_channel_nr // 2),
                                    nn.AvgPool2d(kernel_size=2, stride=2)
                                    )

        self.dec5 = DecoderBlock(bottom_channel_nr + bottom_channel_nr // 2,
                                 bottom_channel_nr,
                                 bottom_channel_nr // 8)

        self.dec4 = DecoderBlock(bottom_channel_nr // 2 + bottom_channel_nr // 8,
                                 bottom_channel_nr // 2,
                                 bottom_channel_nr // 8)
        self.dec3 = DecoderBlock(bottom_channel_nr // 4 + bottom_channel_nr // 8,
                                 bottom_channel_nr // 4,
                                 bottom_channel_nr // 8)
        self.dec2 = DecoderBlock(bottom_channel_nr // 8 + bottom_channel_nr // 8,
                                 bottom_channel_nr // 8,
                                 bottom_channel_nr // 8)
        self.dec1 = DecoderBlock(bottom_channel_nr // 8,
                                 bottom_channel_nr // 16,
                                 bottom_channel_nr // 8)

        if self.use_hypercolumn:
            self.final = nn.Sequential(Conv2dBnRelu(5 * bottom_channel_nr // 8, bottom_channel_nr // 8),
                                       nn.Conv2d(bottom_channel_nr // 8, num_classes, kernel_size=1, padding=0))
        else:
            self.final = nn.Sequential(Conv2dBnRelu(bottom_channel_nr // 8, bottom_channel_nr // 8),
                                       nn.Conv2d(bottom_channel_nr // 8, num_classes, kernel_size=1, padding=0))

    def forward(self, x):
        encoder2, encoder3, encoder4, encoder5 = self.encoders(x)
        encoder5 = F.dropout2d(encoder5, p=self.dropout_2d)

        center = self.center(encoder5)

        dec5 = self.dec5(center, encoder5)
        dec4 = self.dec4(dec5, encoder4)
        dec3 = self.dec3(dec4, encoder3)
        dec2 = self.dec2(dec3, encoder2)
        dec1 = self.dec1(dec2)

        if self.use_hypercolumn:
            dec1 = torch.cat([dec1,
                              F.upsample(dec2, scale_factor=2, mode='bilinear'),
                              F.upsample(dec3, scale_factor=4, mode='bilinear'),
                              F.upsample(dec4, scale_factor=8, mode='bilinear'),
                              F.upsample(dec5, scale_factor=16, mode='bilinear'),
                              ], 1)

        return self.final(dec1)


class UNetSeResNet(nn.Module):
    def __init__(self, encoder_depth, num_classes, dropout_2d=0.0, pretrained=False, use_hypercolumn=False, pool0=False):
        super().__init__()
        self.num_classes = num_classes
        self.dropout_2d = dropout_2d
        self.use_hypercolumn = use_hypercolumn

        self.encoders = SeResNetEncoders(encoder_depth, pretrained=pretrained, pool0=pool0)
        bottom_channel_nr = 2048

        self.center = nn.Sequential(Conv2dBnRelu(bottom_channel_nr, bottom_channel_nr),
                                    Conv2dBnRelu(bottom_channel_nr, bottom_channel_nr // 2),
                                    nn.AvgPool2d(kernel_size=2, stride=2)
                                    )

        self.dec5 = DecoderBlock(bottom_channel_nr + bottom_channel_nr // 2,
                                 bottom_channel_nr,
                                 bottom_channel_nr // 8)

        self.dec4 = DecoderBlock(bottom_channel_nr // 2 + bottom_channel_nr // 8,
                                 bottom_channel_nr // 2,
                                 bottom_channel_nr // 8)
        self.dec3 = DecoderBlock(bottom_channel_nr // 4 + bottom_channel_nr // 8,
                                 bottom_channel_nr // 4,
                                 bottom_channel_nr // 8)
        self.dec2 = DecoderBlock(bottom_channel_nr // 8 + bottom_channel_nr // 8,
                                 bottom_channel_nr // 8,
                                 bottom_channel_nr // 8)
        self.dec1 = DecoderBlock(bottom_channel_nr // 8,
                                 bottom_channel_nr // 16,
                                 bottom_channel_nr // 8)

        if self.use_hypercolumn:
            self.final = nn.Sequential(Conv2dBnRelu(5 * bottom_channel_nr // 8, bottom_channel_nr // 8),
                                       nn.Conv2d(bottom_channel_nr // 8, num_classes, kernel_size=1, padding=0))
        else:
            self.final = nn.Sequential(Conv2dBnRelu(bottom_channel_nr // 8, bottom_channel_nr // 8),
                                       nn.Conv2d(bottom_channel_nr // 8, num_classes, kernel_size=1, padding=0))

    def forward(self, x):
        encoder2, encoder3, encoder4, encoder5 = self.encoders(x)
        encoder5 = F.dropout2d(encoder5, p=self.dropout_2d)

        center = self.center(encoder5)

        dec5 = self.dec5(center, encoder5)
        dec4 = self.dec4(dec5, encoder4)
        dec3 = self.dec3(dec4, encoder3)
        dec2 = self.dec2(dec3, encoder2)
        dec1 = self.dec1(dec2)

        if self.use_hypercolumn:
            dec1 = torch.cat([dec1,
                              F.upsample(dec2, scale_factor=2, mode='bilinear'),
                              F.upsample(dec3, scale_factor=4, mode='bilinear'),
                              F.upsample(dec4, scale_factor=8, mode='bilinear'),
                              F.upsample(dec5, scale_factor=16, mode='bilinear'),
                              ], 1)

        return self.final(dec1)


class UNetSeResNetXt(nn.Module):
    def __init__(self, encoder_depth, num_classes, dropout_2d=0.0, pretrained=False, use_hypercolumn=False, pool0=False):
        super().__init__()
        self.num_classes = num_classes
        self.dropout_2d = dropout_2d
        self.use_hypercolumn = use_hypercolumn

        self.encoders = SeResNetXtEncoders(encoder_depth, pretrained=pretrained, pool0=pool0)
        bottom_channel_nr = 2048

        self.center = nn.Sequential(Conv2dBnRelu(bottom_channel_nr, bottom_channel_nr),
                                    Conv2dBnRelu(bottom_channel_nr, bottom_channel_nr // 2),
                                    nn.AvgPool2d(kernel_size=2, stride=2)
                                    )

        self.dec5 = DecoderBlock(bottom_channel_nr + bottom_channel_nr // 2,
                                 bottom_channel_nr,
                                 bottom_channel_nr // 8)

        self.dec4 = DecoderBlock(bottom_channel_nr // 2 + bottom_channel_nr // 8,
                                 bottom_channel_nr // 2,
                                 bottom_channel_nr // 8)
        self.dec3 = DecoderBlock(bottom_channel_nr // 4 + bottom_channel_nr // 8,
                                 bottom_channel_nr // 4,
                                 bottom_channel_nr // 8)
        self.dec2 = DecoderBlock(bottom_channel_nr // 8 + bottom_channel_nr // 8,
                                 bottom_channel_nr // 8,
                                 bottom_channel_nr // 8)
        self.dec1 = DecoderBlock(bottom_channel_nr // 8,
                                 bottom_channel_nr // 16,
                                 bottom_channel_nr // 8)

        if self.use_hypercolumn:
            self.final = nn.Sequential(Conv2dBnRelu(5 * bottom_channel_nr // 8, bottom_channel_nr // 8),
                                       nn.Conv2d(bottom_channel_nr // 8, num_classes, kernel_size=1, padding=0))
        else:
            self.final = nn.Sequential(Conv2dBnRelu(bottom_channel_nr // 8, bottom_channel_nr // 8),
                                       nn.Conv2d(bottom_channel_nr // 8, num_classes, kernel_size=1, padding=0))

    def forward(self, x):
        encoder2, encoder3, encoder4, encoder5 = self.encoders(x)
        encoder5 = F.dropout2d(encoder5, p=self.dropout_2d)

        center = self.center(encoder5)

        dec5 = self.dec5(center, encoder5)
        dec4 = self.dec4(dec5, encoder4)
        dec3 = self.dec3(dec4, encoder3)
        dec2 = self.dec2(dec3, encoder2)
        dec1 = self.dec1(dec2)

        if self.use_hypercolumn:
            dec1 = torch.cat([dec1,
                              F.upsample(dec2, scale_factor=2, mode='bilinear'),
                              F.upsample(dec3, scale_factor=4, mode='bilinear'),
                              F.upsample(dec4, scale_factor=8, mode='bilinear'),
                              F.upsample(dec5, scale_factor=16, mode='bilinear'),
                              ], 1)

        return self.final(dec1)


class UNetDenseNet(nn.Module):
    def __init__(self, encoder_depth, num_classes, dropout_2d=0.0, pretrained=False, use_hypercolumn=False, pool0=False):
        super().__init__()
        self.num_classes = num_classes
        self.dropout_2d = dropout_2d
        self.use_hypercolumn = use_hypercolumn

        self.encoders = DenseNetEncoders(encoder_depth, pretrained=pretrained, pool0=pool0)
        if encoder_depth == 121:
            encoder_channel_nr = [256, 512, 1024, 1024]
        elif encoder_depth == 161:
            encoder_channel_nr = [384, 768, 2112, 2208]
        elif encoder_depth == 169:
            encoder_channel_nr = [256, 512, 1280, 1664]
        elif encoder_depth == 201:
            encoder_channel_nr = [256, 512, 1792, 1920]
        else:
            raise NotImplementedError('only 121, 161, 169, 201 version of Densenet are implemented')

        self.center = nn.Sequential(Conv2dBnRelu(encoder_channel_nr[3], encoder_channel_nr[3]),
                                    Conv2dBnRelu(encoder_channel_nr[3], encoder_channel_nr[2]),
                                    nn.AvgPool2d(kernel_size=2, stride=2)
                                    )

        self.dec5 = DecoderBlock(encoder_channel_nr[3] + encoder_channel_nr[2],
                                 encoder_channel_nr[3],
                                 encoder_channel_nr[3] // 8)

        self.dec4 = DecoderBlock(encoder_channel_nr[2] + encoder_channel_nr[3] // 8,
                                 encoder_channel_nr[3] // 2,
                                 encoder_channel_nr[3] // 8)
        self.dec3 = DecoderBlock(encoder_channel_nr[1] + encoder_channel_nr[3] // 8,
                                 encoder_channel_nr[3] // 4,
                                 encoder_channel_nr[3] // 8)
        self.dec2 = DecoderBlock(encoder_channel_nr[0] + encoder_channel_nr[3] // 8,
                                 encoder_channel_nr[3] // 8,
                                 encoder_channel_nr[3] // 8)
        self.dec1 = DecoderBlock(encoder_channel_nr[3] // 8,
                                 encoder_channel_nr[3] // 16,
                                 encoder_channel_nr[3] // 8)

        if self.use_hypercolumn:
            self.final = nn.Sequential(Conv2dBnRelu(5 * encoder_channel_nr[3] // 8, encoder_channel_nr[3] // 8),
                                       nn.Conv2d(encoder_channel_nr[3] // 8, num_classes, kernel_size=1, padding=0))
        else:
            self.final = nn.Sequential(Conv2dBnRelu(encoder_channel_nr[3] // 8, encoder_channel_nr[3] // 8),
                                       nn.Conv2d(encoder_channel_nr[3] // 8, num_classes, kernel_size=1, padding=0))

    def forward(self, x):
        encoder2, encoder3, encoder4, encoder5 = self.encoders(x)
        encoder5 = F.dropout2d(encoder5, p=self.dropout_2d)

        center = self.center(encoder5)

        dec5 = self.dec5(center, encoder5)
        dec4 = self.dec4(dec5, encoder4)
        dec3 = self.dec3(dec4, encoder3)
        dec2 = self.dec2(dec3, encoder2)
        dec1 = self.dec1(dec2)

        if self.use_hypercolumn:
            dec1 = torch.cat([dec1,
                              F.upsample(dec2, scale_factor=2, mode='bilinear'),
                              F.upsample(dec3, scale_factor=4, mode='bilinear'),
                              F.upsample(dec4, scale_factor=8, mode='bilinear'),
                              F.upsample(dec5, scale_factor=16, mode='bilinear'),
                              ], 1)

        return self.final(dec1)
