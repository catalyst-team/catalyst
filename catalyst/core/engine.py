from typing import Mapping, Any
from abc import ABC, abstractmethod
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
    def process_model(self):
        pass

    def process_criterion(self):
        pass

    def process_optimizer(self):
        pass

    def process_scheduler(self):
        pass

    def process_components(self):
        pass

    def handle_device(self, batch: Mapping[str, Any]):
        pass
        # return any2device(batch, self.device)

    # taken for utils
    def sync_metric(self):
        pass

    def save_checkpoint(self):
        pass

    def load_checkpoint(self):
        pass
