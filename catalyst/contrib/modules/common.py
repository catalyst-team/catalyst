import torch.nn as nn


class Lambda(nn.Module):
    def __init__(self, lambda_fn):
        super().__init__()
        self.lambda_fn = lambda_fn

    def forward(self, x):
        return self.lambda_fn(x)


class Flatten(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.view(x.shape[0], -1)
