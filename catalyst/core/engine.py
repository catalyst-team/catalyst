# flake8: noqa

from typing import Any, Callable, List, Mapping, Union
from abc import ABC, abstractmethod

import torch
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

    @abstractmethod
    def to_device(
        self, obj: Union[torch.Tensor, nn.Module]
    ) -> Union[torch.Tensor, nn.Module]:
        pass

    def init_process(self):
        # init here process variables in DDP mode
        pass

    def cleanup_process(self):
        # destroy process in DDP mode
        pass

    # @abstractmethod
    def process_components(self):
        pass

    def handle_device(self, batch: Mapping[str, Any]):
        pass
        # return any2device(batch, self.device)

    # taken for utils
    def sync_metric(self) -> None:
        pass

    @abstractmethod
    def save_checkpoint(self) -> None:
        pass

    @abstractmethod
    def load_checkpoint(self) -> None:
        pass

    def zero_grad(self, optimizer: optim.Optimizer) -> None:
        optimizer.zero_grad()

    def optimizer_step(self, optimizer: optim.Optimizer) -> None:
        """Do one optimization step."""
        optimizer.step()

    @property
    @abstractmethod
    def save_fn(self):
        pass
