from typing import Union  # isort:skip

import torch
from torch import nn, optim
from torch.utils import data

Model = nn.Module
Criterion = nn.Module
Optimizer = optim.Optimizer
Scheduler = optim.lr_scheduler._LRScheduler  # noinspection PyProtectedMember
Dataset = data.Dataset
Device = Union[str, torch.device]

__all__ = [
    "Model",
    "Criterion",
    "Optimizer",
    "Scheduler",
    "Dataset",
    "Device",
]
