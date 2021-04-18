"""
All Catalyst custom types are defined in this module.
"""
from typing import Dict, Union
from numbers import Number
from pathlib import Path

import torch
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.utils import data

Directory = Path
File = Path

Model = nn.Module
Criterion = nn.Module
Optimizer = optim.Optimizer
# @TODO: how to fix PyTorch? Union["lr_scheduler._LRScheduler", "lr_scheduler.ReduceLROnPlateau"]
Scheduler = lr_scheduler._LRScheduler  # noqa: WPS437
Dataset = data.Dataset
Device = Union[str, torch.device]

RunnerModel = Union[Model, Dict[str, Model]]
RunnerCriterion = Union[Criterion, Dict[str, Criterion]]
RunnerOptimizer = Union[Optimizer, Dict[str, Optimizer]]
RunnerScheduler = Union[Scheduler, Dict[str, Scheduler]]

__all__ = [
    "Number",
    "Directory",
    "File",
    "Model",
    "Criterion",
    "Optimizer",
    "Scheduler",
    "Dataset",
    "Device",
    "RunnerModel",
    "RunnerCriterion",
    "RunnerOptimizer",
    "RunnerScheduler",
]
