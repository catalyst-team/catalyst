import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models


def get_channels(architecture):
    if architecture in ["resnet18", "resnet34"]:
        return [512, 256, 128, 64]
    elif architecture in ["resnet50", "resnet101", "resnet152"]:
        return [2048, 1024, 512, 256]
    else:
        raise Exception(f"{architecture} is not supported as backbone")


class ConvRelu(nn.Module):
    """3x3 convolution followed by ReLU activation building block.
    """

    def __init__(self, num_in, num_out):
        """Creates a `ConvReLU` building block.

        Args:
          num_in: number of input feature maps
          num_out: number of output feature maps
        """

        super().__init__()

        self.block = nn.Conv2d(
            num_in, num_out, kernel_size=3, padding=1, bias=False
        )

    def forward(self, x):
        """
        The networks forward pass for which
            autograd synthesizes the backwards pass.

        Args:
          x: the input tensor

        Returns:
          The networks output tensor.
        """

        return F.relu(self.block(x), inplace=True)


class DecoderBlock(nn.Module):
    """Decoder building block upsampling resolution by a factor of two.
    """

    def __init__(self, num_in, num_out):
        """Creates a `DecoderBlock` building block.

        Args:
          num_in: number of input feature maps
          num_out: number of output feature maps
        """

        super().__init__()

        self.block = ConvRelu(num_in, num_out)

    def forward(self, x):
        """
        The networks forward pass for which
            autograd synthesizes the backwards pass.

        Args:
          x: the input tensor

        Returns:
          The networks output tensor.
        """

        return self.block(F.interpolate(x, scale_factor=2, mode="nearest"))


class ResNetUnet(nn.Module):
    """
    U-Net inspired encoder-decoder architecture for semantic segmentation,
        with a ResNet encoder as proposed by Alexander Buslaev.

    Also known as AlbuNet

    See:
    - https://arxiv.org/abs/1505.04597 -
        U-Net: Convolutional Networks for Biomedical Image Segmentation
    - https://arxiv.org/abs/1411.4038  -
        Fully Convolutional Networks for Semantic Segmentation
    - https://arxiv.org/abs/1512.03385 -
        Deep Residual Learning for Image Recognition
    - https://arxiv.org/abs/1801.05746 -
        TernausNet: U-Net with VGG11
        Encoder Pre-Trained on ImageNet for Image Segmentation
    - https://arxiv.org/abs/1806.00844 -
        TernausNetV2: Fully Convolutional Network for Instance Segmentation

    based on https://github.com/mapbox/robosat/blob/master/robosat/unet.py

    """
    def __init__(
        self,
        num_classes=1,
        num_filters=32,
        backbone="resnet34",
        pretrained=True
    ):
        """Creates an `UNet` instance for semantic segmentation.

        Args:
          num_classes: number of classes to predict.
          pretrained: use ImageNet pre-trained backbone feature extractor
        """

        super().__init__()

        # Todo: make input channels configurable,
        # not hard-coded to three channels for RGB

        self.resnet = models.__dict__[backbone](pretrained=pretrained)
        encoder_channels = get_channels(backbone)

        # Access resnet directly in forward pass; do not store refs here due to
        # https://github.com/pytorch/pytorch/issues/8392

        self.center = DecoderBlock(encoder_channels[0], num_filters * 8)

        self.dec0 = DecoderBlock(
            num_in=encoder_channels[0] + num_filters * 8,
            num_out=num_filters * 8
        )
        self.dec1 = DecoderBlock(
            num_in=encoder_channels[1] + num_filters * 8,
            num_out=num_filters * 8
        )
        self.dec2 = DecoderBlock(
            num_in=encoder_channels[2] + num_filters * 8,
            num_out=num_filters * 2
        )
        self.dec3 = DecoderBlock(
            num_in=encoder_channels[3] + num_filters * 2,
            num_out=num_filters * 2 * 2
        )
        self.dec4 = DecoderBlock(num_filters * 2 * 2, num_filters)
        self.dec5 = ConvRelu(num_filters, num_filters)

        self.final = nn.Conv2d(num_filters, num_classes, kernel_size=1)

    def forward(self, x):
        """
        The networks forward pass for which
            autograd synthesizes the backwards pass.

        Args:
          x: the input tensor

        Returns:
          The networks output tensor.
        """
        size = x.size()
        assert size[-1] % 64 == 0 and size[-2] % 64 == 0, \
            "image resolution has to be divisible by 64 for resnet"

        enc0 = self.resnet.conv1(x)
        enc0 = self.resnet.bn1(enc0)
        enc0 = self.resnet.relu(enc0)
        enc0 = self.resnet.maxpool(enc0)

        enc1 = self.resnet.layer1(enc0)
        enc2 = self.resnet.layer2(enc1)
        enc3 = self.resnet.layer3(enc2)
        enc4 = self.resnet.layer4(enc3)

        center = self.center(F.max_pool2d(enc4, kernel_size=2, stride=2))

        dec0 = self.dec0(torch.cat([enc4, center], dim=1))
        dec1 = self.dec1(torch.cat([enc3, dec0], dim=1))
        dec2 = self.dec2(torch.cat([enc2, dec1], dim=1))
        dec3 = self.dec3(torch.cat([enc1, dec2], dim=1))
        dec4 = self.dec4(dec3)
        dec5 = self.dec5(dec4)

        return self.final(dec5)
