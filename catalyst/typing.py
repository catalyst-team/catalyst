"""
All Catalyst custom types are defined in this module.
"""
from typing import Dict, Union
from numbers import Number
from pathlib import Path

import torch
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils import data
from torch.utils.data import sampler

Directory = Path
File = Path

TorchModel = nn.Module
TorchCriterion = nn.Module
TorchOptimizer = optim.Optimizer
TorchScheduler = (lr_scheduler._LRScheduler, ReduceLROnPlateau)
TorchDataset = data.Dataset
TorchSampler = sampler.Sampler

RunnerDevice = Union[str, torch.device]
RunnerModel = Union[TorchModel, Dict[str, TorchModel]]
RunnerCriterion = Union[TorchCriterion, Dict[str, TorchCriterion]]
RunnerOptimizer = Union[TorchOptimizer, Dict[str, TorchOptimizer]]
_torch_scheduler = Union[lr_scheduler._LRScheduler, ReduceLROnPlateau]
RunnerScheduler = Union[_torch_scheduler, Dict[str, _torch_scheduler]]

__all__ = [
    "Number",
    "Directory",
    "File",
    "TorchModel",
    "TorchCriterion",
    "TorchOptimizer",
    "TorchScheduler",
    "TorchDataset",
    "TorchSampler",
    "RunnerDevice",
    "RunnerModel",
    "RunnerCriterion",
    "RunnerOptimizer",
    "RunnerScheduler",
]
