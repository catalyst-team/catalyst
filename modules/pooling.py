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
