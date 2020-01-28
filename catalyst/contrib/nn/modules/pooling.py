import torch
import torch.nn as nn
import torch.nn.functional as F

from catalyst.contrib.registry import MODULES


class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        h, w = x.shape[2:]
        return F.avg_pool2d(input=x, kernel_size=(h, w))

    @staticmethod
    def out_features(in_features):
        return in_features


class GlobalMaxPool2d(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        h, w = x.shape[2:]
        return F.max_pool2d(input=x, kernel_size=(h, w))

    @staticmethod
    def out_features(in_features):
        return in_features


class GlobalConcatPool2d(nn.Module):
    def __init__(self):
        super().__init__()
        self.avg = GlobalAvgPool2d()
        self.max = GlobalMaxPool2d()

    def forward(self, x):
        return torch.cat([self.avg(x), self.max(x)], 1)

    @staticmethod
    def out_features(in_features):
        return in_features * 2


class GlobalAttnPool2d(nn.Module):
    def __init__(self, in_features, activation_fn="Sigmoid"):
        super().__init__()

        activation_fn = MODULES.get_if_str(activation_fn)
        self.attn = nn.Sequential(
            nn.Conv2d(
                in_features, 1, kernel_size=1, stride=1, padding=0, bias=False
            ), activation_fn()
        )

    def forward(self, x):
        x_a = self.attn(x)
        x = x * x_a
        x = torch.sum(x, dim=[-2, -1], keepdim=True)
        return x

    @staticmethod
    def out_features(in_features):
        return in_features


class GlobalAvgAttnPool2d(nn.Module):
    def __init__(self, in_features, activation_fn="Sigmoid"):
        super().__init__()
        self.avg = GlobalAvgPool2d()
        self.attn = GlobalAttnPool2d(in_features, activation_fn)

    def forward(self, x):
        return torch.cat([self.avg(x), self.attn(x)], 1)

    @staticmethod
    def out_features(in_features):
        return in_features * 2


class GlobalMaxAttnPool2d(nn.Module):
    def __init__(self, in_features, activation_fn="Sigmoid"):
        super().__init__()
        self.max = GlobalMaxPool2d()
        self.attn = GlobalAttnPool2d(in_features, activation_fn)

    def forward(self, x):
        return torch.cat([self.max(x), self.attn(x)], 1)

    @staticmethod
    def out_features(in_features):
        return in_features * 2


class GlobalConcatAttnPool2d(nn.Module):
    def __init__(self, in_features, activation_fn="Sigmoid"):
        super().__init__()
        self.avg = GlobalAvgPool2d()
        self.max = GlobalMaxPool2d()
        self.attn = GlobalAttnPool2d(in_features, activation_fn)

    def forward(self, x):
        return torch.cat([self.avg(x), self.max(x), self.attn(x)], 1)

    @staticmethod
    def out_features(in_features):
        return in_features * 3
