# flake8: noqa

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet, resnet

__all__ = ("SingleShotDetector", "CenterNet")

_channels_map = {
    "resnet18": [256, 512, 512, 256, 256, 128],
    "resnet34": [256, 512, 512, 256, 256, 256],
    "resnet50": [1024, 512, 512, 256, 256, 256],
    "resnet101": [1024, 512, 512, 256, 256, 256],
    "resnet152": [1024, 512, 512, 256, 256, 256],
}

_backbones = {
    "resnet18": (resnet.resnet18, 512),
    "resnet34": (resnet.resnet34, 512),
    "resnet50": (resnet.resnet50, 2048),
    "resnet101": (resnet.resnet101, 2048),
    "resnet152": (resnet.resnet152, 2048),
    "mobilenet_v2": (mobilenet.mobilenet_v2, 1280)
    # "mobilenet_v3_small": (torchvision.models.mobilenet_v3_small, 576),
    # "mobilenet_v3_large": (torchvision.models.mobilenet_v3_large, 960),
}


class ResnetBackbone(nn.Module):
    def __init__(self, backbone="resnet50", backbone_path=None):
        """
        Args:
            backbone (str): resnet backbone to use.
                Expected one of ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]
                Default is "resnet50".
            backbone_path (str): path to pretrained backbone model.
                If ``None`` then will be used torchvision pretrained model.
                Default is None.
        """
        super().__init__()

        self.out_channels = _channels_map[backbone]
        if backbone == "resnet18":
            backbone = resnet.resnet18(pretrained=not backbone_path)
        elif backbone == "resnet34":
            backbone = resnet.resnet34(pretrained=not backbone_path)
        elif backbone == "resnet50":
            backbone = resnet.resnet50(pretrained=not backbone_path)
        elif backbone == "resnet101":
            backbone = resnet.resnet101(pretrained=not backbone_path)
        elif backbone == "resnet152":
            backbone = resnet.resnet152(pretrained=not backbone_path)
        else:
            raise ValueError(f"Unknown ResNet backbone - '{backbone}'!")

        if backbone_path:
            backbone.load_state_dict(torch.load(backbone_path))

        self.feature_extractor = nn.Sequential(*list(backbone.children())[:7])

        conv4_block1 = self.feature_extractor[-1][0]

        conv4_block1.conv1.stride = (1, 1)
        conv4_block1.conv2.stride = (1, 1)
        conv4_block1.downsample[0].stride = (1, 1)

    def forward(self, x):
        x = self.feature_extractor(x)
        return x


class SingleShotDetector(nn.Module):
    def __init__(self, backbone="resnet18", num_classes=80):
        """
        Source:
            https://github.com/NVIDIA/DeepLearningExamples/blob/70fcb70ff4bc49cc723195b35cfa8d4ce94a7f76/PyTorch/Detection/SSD/src/model.py

        Args:
            backbone (str): model backbone to use
            n_classes (int): number of classes to predict
        """
        super().__init__()

        self.feature_extractor = ResnetBackbone(backbone)

        self.label_num = num_classes + 1  # +background class
        self._build_additional_features(self.feature_extractor.out_channels)
        self.num_defaults = [4, 6, 6, 6, 4, 4]
        self.loc = []
        self.conf = []

        for nd, oc in zip(self.num_defaults, self.feature_extractor.out_channels):
            self.loc.append(nn.Conv2d(oc, nd * 4, kernel_size=3, padding=1))
            self.conf.append(nn.Conv2d(oc, nd * self.label_num, kernel_size=3, padding=1))

        self.loc = nn.ModuleList(self.loc)
        self.conf = nn.ModuleList(self.conf)
        self._init_weights()

    def _build_additional_features(self, input_size):
        self.additional_blocks = []
        for i, (input_size, output_size, channels) in enumerate(
            zip(input_size[:-1], input_size[1:], [256, 256, 128, 128, 128])
        ):
            if i < 3:
                layer = nn.Sequential(
                    nn.Conv2d(input_size, channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(
                        channels, output_size, kernel_size=3, padding=1, stride=2, bias=False
                    ),
                    nn.BatchNorm2d(output_size),
                    nn.ReLU(inplace=True),
                )
            else:
                layer = nn.Sequential(
                    nn.Conv2d(input_size, channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(channels, output_size, kernel_size=3, bias=False),
                    nn.BatchNorm2d(output_size),
                    nn.ReLU(inplace=True),
                )

            self.additional_blocks.append(layer)

        self.additional_blocks = nn.ModuleList(self.additional_blocks)

    def _init_weights(self):
        layers = [*self.additional_blocks, *self.loc, *self.conf]
        for layer in layers:
            for param in layer.parameters():
                if param.dim() > 1:
                    nn.init.xavier_uniform_(param)

    # Shape the classifier to the view of bboxes
    def bbox_view(self, src, loc, conf):
        ret = []
        for s, l, c in zip(src, loc, conf):
            # ret.append((l(s).view(s.size(0), 4, -1), c(s).view(s.size(0), self.label_num, -1)))
            ret.append((l(s).view(s.size(0), -1, 4), c(s).view(s.size(0), -1, self.label_num)))

        locs, confs = list(zip(*ret))
        # locs, confs = torch.cat(locs, 2).contiguous(), torch.cat(confs, 2).contiguous()
        locs, confs = torch.cat(locs, 1).contiguous(), torch.cat(confs, 1).contiguous()
        return locs, confs

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): batch of data, expected shapes [B, 3, H, W]

        Returns:
            bbox locations (torch.Tensor) with shapes [B, A, 4],
                where B - batch size, A - num anchors
            class confidence logits (torch.Tensor) with shapes [B, A, N_CLASSES],
                where B - batch size, A - num anchors
        """
        x = self.feature_extractor(x)

        detection_feed = [x]
        for layer in self.additional_blocks:
            x = layer(x)
            detection_feed.append(x)

        # Feature Map 38x38x4, 19x19x6, 10x10x6, 5x5x6, 3x3x4, 1x1x4
        locs, confs = self.bbox_view(detection_feed, self.loc, self.conf)

        return locs, confs


class DoubleConv(nn.Module):
    """(conv => BN => ReLU) * 2"""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class Interpolate(nn.Module):
    def __init__(
        self,
        size=None,
        scale_factor=None,
        mode="nearest",
        align_corners=None,
        recompute_scale_factor=None,
    ):
        super().__init__()
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners
        self.recompute_scale_factor = recompute_scale_factor

    def forward(self, inputs):
        return F.interpolate(
            inputs,
            size=self.size,
            scale_factor=self.scale_factor,
            mode=self.mode,
            align_corners=self.align_corners,
            recompute_scale_factor=self.recompute_scale_factor,
        )


class UpDoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mode=None):
        super().__init__()
        self.in_ch = in_channels
        self.out_ch = out_channels
        self.mode = mode
        if mode is None:
            self.up = nn.ConvTranspose2d(in_channels, in_channels, 2, stride=2)
        else:
            align_corners = None if mode == "nearest" else True
            self.up = Interpolate(scale_factor=2, mode=mode, align_corners=align_corners)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2=None):
        x1 = self.up(x1)

        if x2 is not None:
            x = torch.cat([x2, x1], dim=1)
            # input is CHW
            diffY = x2.size()[2] - x1.size()[2]
            diffX = x2.size()[3] - x1.size()[3]
            x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2))
        else:
            x = x1

        x = self.conv(x)
        return x


class CenterNet(nn.Module):
    def __init__(self, num_classes=1, backbone="resnet18", upsample_mode="nearest"):
        super().__init__()
        # create backbone.
        basemodel = _backbones[backbone][0](pretrained=True)
        if backbone == "mobilenet_v2":
            layers = list(basemodel.children())[:-1]
        else:
            layers = list(basemodel.children())[:-2]
        basemodel = nn.Sequential(*layers)
        # set basemodel
        self.base_model = basemodel
        self.upsample_mode = upsample_mode

        num_ch = _backbones[backbone][1]

        # original upsample mode was "bilinear"
        self.up1 = UpDoubleConv(num_ch, 512, upsample_mode)
        self.up2 = UpDoubleConv(512, 256, upsample_mode)
        self.up3 = UpDoubleConv(256, 256, upsample_mode)
        # output classification
        self.out_classification = nn.Conv2d(256, num_classes, 1)
        # output residue
        self.out_residue = nn.Conv2d(256, 2, 1)

    def forward(self, x):
        x = self.base_model(x)
        # Add positional info
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        c = self.out_classification(x)  # NOTE: do not forget to apply sigmoid to obtain scores!
        r = self.out_residue(x)
        return c, r
