"""
All Catalyst custom types are defined in this module.
"""
from typing import Union

import torch
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.utils import data

Model = nn.Module
Criterion = nn.Module
Optimizer = optim.Optimizer
Scheduler = lr_scheduler._LRScheduler  # noinspection PyProtectedMember
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
