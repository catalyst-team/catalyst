import torch
import torch.nn as nn
from torchvision import models


class ResNetEncoder(nn.Module):
    def __init__(self, arch, pretrained=True):
        super().__init__()

        backbone = arch(pretrained=pretrained)

        self.encoder0 = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool
        )
        self.encoder1 = backbone.layer1
        self.encoder2 = backbone.layer2
        self.encoder3 = backbone.layer3
        self.encoder4 = backbone.layer4

        self.filters = [
            self.encoder1[-1].conv2.out_channels,
            self.encoder2[-1].conv2.out_channels,
            self.encoder3[-1].conv2.out_channels,
            self.encoder4[-1].conv2.out_channels
        ]

    def forward(self, x):
        acts = []
        x = self.encoder0(x)
        x = self.encoder1(x)
        acts.append(x)
        x = self.encoder2(x)
        acts.append(x)
        x = self.encoder3(x)
        acts.append(x)
        x = self.encoder4(x)
        acts.append(x)
        return acts


class DecoderBlock(nn.Module):
    def __init__(self, m, n, stride=2):
        super().__init__()

        # B, C, H, W -> B, C/4, H, W
        self.conv1 = nn.Conv2d(m, m // 4, 1)
        self.norm1 = nn.BatchNorm2d(m // 4)
        self.relu1 = nn.ReLU(inplace=True)

        # B, C/4, H, W -> B, C/4, H, W
        self.conv2 = nn.ConvTranspose2d(
            m // 4, m // 4, 3, stride=stride, padding=1
        )
        self.norm2 = nn.BatchNorm2d(m // 4)
        self.relu2 = nn.ReLU(inplace=True)

        # B, C/4, H, W -> B, C, H, W
        self.conv3 = nn.Conv2d(m // 4, n, 1)
        self.norm3 = nn.BatchNorm2d(n)
        self.relu3 = nn.ReLU(inplace=True)

    def forward(self, x):
        double_size = (x.size(-2) * 2, x.size(-1) * 2)
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.conv2(x, output_size=double_size)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x


class FinalBlock(nn.Module):
    def __init__(self, num_filters, num_classes=2):
        super().__init__()

        self.conv1 = nn.ConvTranspose2d(
            num_filters, num_filters // 2, 3, stride=2, padding=1
        )
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            num_filters // 2, num_filters // 2, 3, padding=1
        )
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(num_filters // 2, num_classes, 1)

    def forward(self, inputs):
        double_size = (inputs.size(-2) * 2, inputs.size(-1) * 2)
        x = self.conv1(inputs, output_size=double_size)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        return x


class LinkNet(nn.Module):
    def __init__(self, num_classes=1, backbone="resnet34", pretrained=True):
        super().__init__()

        backbone = str(backbone)
        if backbone == "resnet18":
            self.encoder = ResNetEncoder(
                models.resnet18, pretrained=pretrained
            )
        elif backbone == "resnet34":
            self.encoder = ResNetEncoder(
                models.resnet34, pretrained=pretrained
            )
        elif backbone == "resnet50":
            self.encoder = ResNetEncoder(
                models.resnet50, pretrained=pretrained
            )
        else:
            raise ValueError(f"Unexpected LinkNet backbone: {backbone}")
        filters = self.encoder.filters

        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.final = FinalBlock(filters[0], num_classes)

    def forward(self, x):
        e1, e2, e3, e4 = self.encoder(x)

        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)

        return self.final(d1)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = LinkNet(1).to(device)
    images = torch.randn(4, 3, 256, 256).to(device)

    out = model.forward(images)
    print(out.size())
