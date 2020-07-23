# flake8: noqa
import torch
from torch import nn
from torch.nn.functional import relu

from catalyst.contrib.nn.modules.common import Normalize


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 2)
        self.norm = Normalize()

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = relu(self.fc1(x))
        return x
