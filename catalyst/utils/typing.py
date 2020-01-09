from typing import Union, Dict  # isort:skip

import torch
from torch import nn, optim
from torch.utils import data

Model = Union[nn.Module, Dict[str, nn.Module]]
Criterion = nn.Module
Optimizer = Union[optim.Adam, optim.SGD]
Scheduler = optim.lr_scheduler._LRScheduler  # noinspection PyProtectedMember
Dataset = data.Dataset
DataLoader = data.DataLoader
Device = Union[str, torch.device]

__all__ = [
    "Model",
    "Criterion",
    "Optimizer",
    "Scheduler",
    "Device",
    "Dataset",
    "DataLoader"
]
