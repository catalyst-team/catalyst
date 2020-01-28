from typing import Union  # isort:skip

import torch
from torch import nn, optim

Model = nn.Module
Criterion = nn.Module
Optimizer = optim.Optimizer
Scheduler = optim.lr_scheduler._LRScheduler  # noinspection PyProtectedMember
Device = Union[str, torch.device]

__all__ = ["Model", "Criterion", "Optimizer", "Scheduler", "Device"]
