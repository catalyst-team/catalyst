import torch
import torch.nn as nn


class AdaptiveConcatPool2d(nn.Module):
    def __init__(self, sz=None):
        super().__init__()
        sz = sz or (1, 1)
        self.ap = nn.AdaptiveAvgPool2d(sz)
        self.mp = nn.AdaptiveMaxPool2d(sz)

    def forward(self, x):
        return torch.cat([self.mp(x), self.ap(x)], 1)


class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        h, w = x.shape[2:]
        return nn.functional.avg_pool2d(
            input=x,
            kernel_size=(h, w))


class GlobalMaxPool2d(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        h, w = x.shape[2:]
        return nn.functional.max_pool2d(
            input=x,
            kernel_size=(h, w))


class GlobalConcatPool2d(nn.Module):
    def __init__(self):
        super().__init__()
        self.avg = GlobalAvgPool2d()
        self.max = GlobalMaxPool2d()

    def forward(self, x):
        return torch.cat([self.avg(x), self.max(x)], 1)


class GlobalAvgPool1d(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # xx = x.unsqueeze_(-1)
        return nn.functional.avg_pool1d(
            input=x,
            kernel_size=x.shape[1],
            padding=x.shape[1]//2
        )


class GlobalMaxPool1d(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # xx = x.unsqueeze_(-1)
        return nn.functional.max_pool1d(
            input=x,
            kernel_size=x.shape[1],
            padding=x.shape[1]//2
        )


class GlobalConcatPool1d(nn.Module):
    def __init__(self):
        super().__init__()
        self.avg = GlobalAvgPool1d()
        self.max = GlobalMaxPool1d()

    def forward(self, x):
        x = x.unsqueeze_(-1)
        return torch.cat([self.avg(x), self.max(x)], 1)

class GlobalAttnPool2d(nn.Module):
    def __init__(self, in_features, activation_fn="Softmax2d"):
        super().__init__()
        # hack to prevent cycle imports
        from catalyst.modules.modules import name2nn
        activation_fn = name2nn(activation_fn)
        self.attn = nn.Sequential(
            nn.Conv2d(
                in_features, 1,
                kernel_size=1, stride=1,
                padding=0, bias=False),
            activation_fn())

    def forward(self, x):
        h, w = x.shape[2:]
        x_a = self.attn(x)
        x = x * x_a
        return nn.functional.avg_pool2d(
            input=x,
            kernel_size=(h, w))


class GlobalAvgAttnPool2d(nn.Module):
    def __init__(self, in_features, activation_fn="Softmax2d"):
        super().__init__()
        self.avg = GlobalAvgPool2d()
        self.attn = GlobalAttnPool2d(in_features, activation_fn)

    def forward(self, x):
        return torch.cat([self.avg(x), self.attn(x)], 1)


class GlobalMaxAttnPool2d(nn.Module):
    def __init__(self, in_features, activation_fn="Softmax2d"):
        super().__init__()
        self.max = GlobalMaxPool2d()
        self.attn = GlobalAttnPool2d(in_features, activation_fn)

    def forward(self, x):
        return torch.cat([self.max(x), self.attn(x)], 1)


class GlobalConcatAttnPool2d(nn.Module):
    def __init__(self, in_features, activation_fn="Softmax2d"):
        super().__init__()
        self.avg = GlobalAvgPool2d()
        self.max = GlobalMaxPool2d()
        self.attn = GlobalAttnPool2d(in_features, activation_fn)

    def forward(self, x):
        return torch.cat([self.avg(x), self.max(x), self.attn(x)], 1)
