from torch import nn
from torch.nn import functional as F
import torch
from torchvision import models
import torchvision


def conv3x3(in_, out):
    return nn.Conv2d(in_, out, 3, padding=1)


class ConvRelu(nn.Module):
    def __init__(self, in_, out):
        super().__init__()
        self.conv = conv3x3(in_, out)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            ConvRelu(in_channels, middle_channels),
            nn.ConvTranspose2d(
                middle_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class DecoderBlockV2(nn.Module):
    def __init__(
            self, in_channels, middle_channels, out_channels,
            is_deconv=True):
        super(DecoderBlockV2, self).__init__()
        self.in_channels = in_channels

        if is_deconv:
            """
                Paramaters for Deconvolution were chosen to avoid artifacts,
                following link https://distill.pub/2016/deconv-checkerboard/
            """

            self.block = nn.Sequential(
                ConvRelu(in_channels, middle_channels),
                nn.ConvTranspose2d(
                    middle_channels,
                    out_channels,
                    kernel_size=4,
                    stride=2,
                    padding=1),
                nn.ReLU(inplace=True)
            )
        else:
            self.block = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear"),
                ConvRelu(in_channels, middle_channels),
                ConvRelu(middle_channels, out_channels),
            )

    def forward(self, x):
        return self.block(x)


class UNetVgg11(nn.Module):
    def __init__(self, num_filters=32, pretrained=False):
        """
        :param num_classes:
        :param num_filters:
        :param pretrained:
            False - no pre-trained network used
            True - encoder pre-trained with VGG11
        """
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)

        self.encoder = models.vgg11(pretrained=pretrained).features

        self.relu = self.encoder[1]
        self.conv1 = self.encoder[0]
        self.conv2 = self.encoder[3]
        self.conv3s = self.encoder[6]
        self.conv3 = self.encoder[8]
        self.conv4s = self.encoder[11]
        self.conv4 = self.encoder[13]
        self.conv5s = self.encoder[16]
        self.conv5 = self.encoder[18]

        self.center = DecoderBlock(
            num_filters * 8 * 2,
            num_filters * 8 * 2,
            num_filters * 8)
        self.dec5 = DecoderBlock(
            num_filters * (16 + 8),
            num_filters * 8 * 2,
            num_filters * 8)
        self.dec4 = DecoderBlock(
            num_filters * (16 + 8),
            num_filters * 8 * 2,
            num_filters * 4)
        self.dec3 = DecoderBlock(
            num_filters * (8 + 4),
            num_filters * 4 * 2,
            num_filters * 2)
        self.dec2 = DecoderBlock(
            num_filters * (4 + 2),
            num_filters * 2 * 2,
            num_filters)
        self.dec1 = ConvRelu(num_filters * (2 + 1), num_filters)

        self.final = nn.Conv2d(num_filters, 1, kernel_size=1)

    def forward(self, x):
        conv1 = self.relu(self.conv1(x))
        conv2 = self.relu(self.conv2(self.pool(conv1)))
        conv3s = self.relu(self.conv3s(self.pool(conv2)))
        conv3 = self.relu(self.conv3(conv3s))
        conv4s = self.relu(self.conv4s(self.pool(conv3)))
        conv4 = self.relu(self.conv4(conv4s))
        conv5s = self.relu(self.conv5s(self.pool(conv4)))
        conv5 = self.relu(self.conv5(conv5s))

        center = self.center(self.pool(conv5))

        dec5 = self.dec5(torch.cat([center, conv5], 1))
        dec4 = self.dec4(torch.cat([dec5, conv4], 1))
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        dec1 = self.dec1(torch.cat([dec2, conv1], 1))
        return self.final(dec1)


class UNetVgg16(nn.Module):
    def __init__(
            self,
            num_classes=1,
            num_filters=32,
            pretrained=False,
            is_deconv=False):
        """
        :param num_classes:
        :param num_filters:
        :param pretrained:
            False - no pre-trained network used
            True - encoder pre-trained with VGG16
        :is_deconv:
            False: bilinear interpolation is used in decoder
            True: deconvolution is used in decoder
        """
        super().__init__()
        self.num_classes = num_classes

        self.pool = nn.MaxPool2d(2, 2)

        self.encoder = torchvision.models.vgg16(pretrained=pretrained).features

        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Sequential(
            self.encoder[0],
            self.relu,
            self.encoder[2],
            self.relu)

        self.conv2 = nn.Sequential(
            self.encoder[5],
            self.relu,
            self.encoder[7],
            self.relu)

        self.conv3 = nn.Sequential(
            self.encoder[10],
            self.relu,
            self.encoder[12],
            self.relu,
            self.encoder[14],
            self.relu)

        self.conv4 = nn.Sequential(
            self.encoder[17],
            self.relu,
            self.encoder[19],
            self.relu,
            self.encoder[21],
            self.relu)

        self.conv5 = nn.Sequential(
            self.encoder[24],
            self.relu,
            self.encoder[26],
            self.relu,
            self.encoder[28],
            self.relu)

        self.center = DecoderBlockV2(
            512,
            num_filters * 8 * 2,
            num_filters * 8,
            is_deconv)

        self.dec5 = DecoderBlockV2(
            512 + num_filters * 8,
            num_filters * 8 * 2,
            num_filters * 8,
            is_deconv)
        self.dec4 = DecoderBlockV2(
            512 + num_filters * 8,
            num_filters * 8 * 2,
            num_filters * 8,
            is_deconv)
        self.dec3 = DecoderBlockV2(
            256 + num_filters * 8,
            num_filters * 4 * 2,
            num_filters * 2,
            is_deconv)
        self.dec2 = DecoderBlockV2(
            128 + num_filters * 2,
            num_filters * 2 * 2,
            num_filters,
            is_deconv)
        self.dec1 = ConvRelu(64 + num_filters, num_filters)
        self.final = nn.Conv2d(num_filters, num_classes, kernel_size=1)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(self.pool(conv1))
        conv3 = self.conv3(self.pool(conv2))
        conv4 = self.conv4(self.pool(conv3))
        conv5 = self.conv5(self.pool(conv4))

        center = self.center(self.pool(conv5))

        dec5 = self.dec5(torch.cat([center, conv5], 1))

        dec4 = self.dec4(torch.cat([dec5, conv4], 1))
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        dec1 = self.dec1(torch.cat([dec2, conv1], 1))

        if self.num_classes > 1:
            x_out = F.log_softmax(self.final(dec1), dim=1)
        else:
            x_out = F.sigmoid(self.final(dec1))

        return x_out


class UNetRes34(nn.Module):
    """
    UNet (https://arxiv.org/abs/1505.04597)
        with Resnet34(https://arxiv.org/abs/1512.03385) encoder

    Proposed by Alexander Buslaev: https://www.linkedin.com/in/al-buslaev/

    """

    def __init__(
            self,
            num_classes=1,
            num_filters=32,
            pretrained=False,
            is_deconv=False):
        """
        :param num_classes:
        :param num_filters:
        :param pretrained:
            False - no pre-trained network used
            True - encoder pre-trained with resnet34
        :is_deconv:
            False: bilinear interpolation is used in decoder
            True: deconvolution is used in decoder
        """
        super().__init__()
        self.num_classes = num_classes

        self.pool = nn.MaxPool2d(2, 2)

        self.encoder = torchvision.models.resnet34(pretrained=pretrained)

        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Sequential(
            self.encoder.conv1,
            self.encoder.bn1,
            self.encoder.relu,
            self.pool)

        self.conv2 = self.encoder.layer1

        self.conv3 = self.encoder.layer2

        self.conv4 = self.encoder.layer3

        self.conv5 = self.encoder.layer4

        self.center = DecoderBlockV2(
            512,
            num_filters * 8 * 2,
            num_filters * 8,
            is_deconv)

        self.dec5 = DecoderBlockV2(
            512 + num_filters * 8,
            num_filters * 8 * 2,
            num_filters * 8,
            is_deconv)
        self.dec4 = DecoderBlockV2(
            256 + num_filters * 8,
            num_filters * 8 * 2,
            num_filters * 8,
            is_deconv)
        self.dec3 = DecoderBlockV2(
            128 + num_filters * 8,
            num_filters * 4 * 2,
            num_filters * 2,
            is_deconv)
        self.dec2 = DecoderBlockV2(
            64 + num_filters * 2,
            num_filters * 2 * 2,
            num_filters * 2 * 2,
            is_deconv)
        self.dec1 = DecoderBlockV2(
            num_filters * 2 * 2,
            num_filters * 2 * 2,
            num_filters,
            is_deconv)
        self.dec0 = ConvRelu(num_filters, num_filters)
        self.final = nn.Conv2d(num_filters, num_classes, kernel_size=1)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        center = self.center(self.pool(conv5))

        dec5 = self.dec5(torch.cat([center, conv5], 1))

        dec4 = self.dec4(torch.cat([dec5, conv4], 1))
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        dec1 = self.dec1(dec2)
        dec0 = self.dec0(dec1)

        if self.num_classes > 1:
            x_out = F.log_softmax(self.final(dec0), dim=1)
        else:
            x_out = F.sigmoid(self.final(dec0))

        return x_out
