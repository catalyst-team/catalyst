# flake8: noqa

from typing import Any, Callable, List, Mapping
from abc import ABC, abstractmethod

import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from catalyst.core.callback import ICallback


class IEngine(ABC, ICallback):
    """
    An abstraction that wraps torch.device
    for different hardware-specific configurations.

    - cpu
    - single-gpu
    - multi-gpu
    - amp (nvidia, torch)
    - ddp (torch, etc)
    """

    # taken for runner
    def process_model(self, model: nn.Module) -> nn.Module:
        pass

    def process_criterion(self, criterion: nn.Module) -> nn.Module:
        pass

    def process_optimizer(
        self, optimizer: optim.Optimizer, model: nn.Module
    ) -> optim.Optimizer:
        pass

    def process_scheduler(
        self, scheduler: nn.Module, optimizer: optim.Optimizer
    ) -> nn.Module:
        pass

    def process_components(self):
        pass

    def handle_device(self, batch: Mapping[str, Any]):
        pass
        # return any2device(batch, self.device)

    # taken for utils
    def sync_metric(self) -> None:
        pass

    def save_checkpoint(self) -> None:
        pass

    def load_checkpoint(self) -> None:
        pass

    def zero_grad(self, optimizer: optim.Optimizer) -> None:
        optimizer.zero_grad()

    def optimizer_step(self, optimizer: optim.Optimizer) -> None:
        """Do one optimization step."""
        optimizer.step()
