import torch
from torch import nn


def conv3x3(in_channels, out_channels, dilation=1):
    return nn.Conv2d(
        in_channels, out_channels, 3, padding=dilation, dilation=dilation
    )


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, batch_norm=False):
        super().__init__()

        self.block = nn.Sequential()
        self.block.add_module("conv1", conv3x3(in_channels, out_channels))
        if batch_norm:
            self.block.add_module("bn1", nn.BatchNorm2d(out_channels))
        self.block.add_module("relu1", nn.ReLU())
        self.block.add_module("conv2", conv3x3(out_channels, out_channels))
        if batch_norm:
            self.block.add_module("bn2", nn.BatchNorm2d(out_channels))
        self.block.add_module("relu2", nn.ReLU())

    def forward(self, x):
        return self.block(x)


class Encoder(nn.Module):
    def __init__(self, in_channels, num_filters, num_blocks):
        super().__init__()

        self.num_blocks = num_blocks
        for i in range(num_blocks):
            in_channels = in_channels if not i else num_filters * 2**(i - 1)
            out_channels = num_filters * 2**i
            self.add_module(
                f"block{i + 1}", EncoderBlock(in_channels, out_channels)
            )
            if i != num_blocks - 1:
                self.add_module(f"pool{i + 1}", nn.MaxPool2d(2, 2))

    def forward(self, x):
        acts = []
        for i in range(self.num_blocks):
            x = self.__getattr__(f"block{i + 1}")(x)
            acts.append(x)
            if i != self.num_blocks - 1:
                x = self.__getattr__(f"pool{i + 1}")(x)
        return acts


class DecoderBlock(nn.Module):
    def __init__(self, out_channels):
        super().__init__()

        self.uppool = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=True
        )
        self.upconv = conv3x3(out_channels * 2, out_channels)
        self.conv1 = conv3x3(out_channels * 2, out_channels)
        self.conv2 = conv3x3(out_channels, out_channels)

    def forward(self, down, left):
        x = self.uppool(down)
        x = self.upconv(x)
        x = torch.cat([left, x], 1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class Decoder(nn.Module):
    def __init__(self, num_filters, num_blocks):
        super().__init__()

        for i in range(num_blocks):
            self.add_module(
                f"block{num_blocks - i}", DecoderBlock(num_filters * 2**i)
            )

    def forward(self, acts):
        up = acts[-1]
        for i, left in enumerate(acts[-2::-1]):
            up = self.__getattr__(f"block{i + 1}")(up, left)
        return up


class UNet(nn.Module):
    """
    CNN architecture for semantic segmentation
    Made by @nizhib
    """
    def __init__(
        self, num_classes=1, in_channels=3, num_filters=64, num_blocks=4
    ):
        super().__init__()

        self.encoder = Encoder(in_channels, num_filters, num_blocks)
        self.decoder = Decoder(num_filters, num_blocks - 1)
        self.final = nn.Conv2d(num_filters, num_classes, 1)

    def forward(self, x):
        acts = self.encoder(x)
        x = self.decoder(acts)
        x = self.final(x)
        return x


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNet(num_classes=1).to(device)
    images = torch.randn(4, 3, 256, 256).to(device)

    out = model.forward(images)
    print(out.size())
