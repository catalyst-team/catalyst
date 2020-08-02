"""
All Catalyst custom types are defined in this module.
"""
from typing import Dict, Union

import torch
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.utils import data

Model = nn.Module
Criterion = nn.Module
Optimizer = optim.Optimizer
Scheduler = lr_scheduler._LRScheduler  # noqa: WPS437
Dataset = data.Dataset
Device = Union[str, torch.device]

RunnerModel = Union[Model, Dict[str, Model]]
RunnerCriterion = Union[Criterion, Dict[str, Criterion]]
RunnerOptimizer = Union[Optimizer, Dict[str, Optimizer]]
RunnerScheduler = Union[Scheduler, Dict[str, Scheduler]]

__all__ = [
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
