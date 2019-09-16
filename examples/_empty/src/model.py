import torch
from catalyst.contrib import registry


@registry.Model
class Model(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, x):
        # CHANGE ME
        return x
