from typing import Callable

from catalyst.engines.torch import DeviceEngine
from catalyst.settings import SETTINGS

if SETTINGS.xla_required:
    import torch_xla.core.xla_model as xm


class XLAEngine(DeviceEngine):
    def __init__(self):
        """Init."""
        super().__init__()
        self._device = xm.xla_device()

    def ddp_sync_run(self, function: Callable):
        function()

    def optimizer_step(self, loss, model, optimizer) -> None:
        """Abstraction over ``optimizer.step()`` step."""
        xm.optimizer_step(optimizer, barrier=True)


class DistributedXLAEngine(DeviceEngine):
    def __init__(self):
        """Init."""
        super().__init__()
        self._device = None
        self._rank = 0
        self._world_size = 8

    @property
    def is_xla_ddp(self) -> bool:
        """Boolean flag for XLA distributed run."""
        return True

    @property
    def rank(self) -> int:
        """Process rank for distributed training."""
        return self._rank

    @property
    def world_size(self) -> int:
        """Process world size  for distributed training."""
        return self._world_size

    def setup_process(self, rank: int = -1, world_size: int = 1):
        """Initialize DDP variables and processes.

        Args:
            rank: process rank. Default is `-1`.
            world_size: number of devices in netwok to expect for train. Default is `1`.
        """
        self._rank = rank
        self._world_size = world_size
        self._device = xm.xla_device()

    def ddp_sync_run(self, function: Callable):
        if self.rank > 0:
            xm.rendezvous("ddp_sync_run")
        function()
        if self.rank == 0:
            xm.rendezvous("ddp_sync_run")

    def optimizer_step(self, loss, model, optimizer) -> None:
        """Abstraction over ``optimizer.step()`` step."""
        xm.optimizer_step(optimizer)


__all__ = ["XLAEngine", "DistributedXLAEngine"]
