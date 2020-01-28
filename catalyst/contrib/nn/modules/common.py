import torch.nn as nn
import torch.nn.functional as F


class Flatten(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.view(x.shape[0], -1)


class Lambda(nn.Module):
    def __init__(self, lambda_fn):
        super().__init__()
        self.lambda_fn = lambda_fn

    def forward(self, x):
        return self.lambda_fn(x)


class Normalize(nn.Module):
    def __init__(self, **normalize_kwargs):
        super().__init__()
        self.normalize_kwargs = normalize_kwargs

    def forward(self, x):
        return F.normalize(x, **self.normalize_kwargs)
