from typing import Any, Callable, Dict, Optional

import numpy as np
import torch

from catalyst.engines.torch import DeviceEngine
from catalyst.settings import SETTINGS

if SETTINGS.xla_required:
    import torch_xla.core.xla_model as xm
    from torch_xla.distributed.parallel_loader import ParallelLoader
    import torch_xla.distributed.xla_multiprocessing as xmp


class XLAEngine(DeviceEngine):
    def __init__(self):
        """Init."""
        super().__init__()
        self._device = xm.xla_device()

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
        self._backend = "xla"

    @property
    def rank(self) -> int:
        """Process rank for distributed training."""
        return self._rank

    @property
    def world_size(self) -> int:
        """Process world size  for distributed training."""
        return self._world_size

    @property
    def backend(self) -> Optional[str]:
        return self._backend

    def barrier(self) -> None:
        xm.rendezvous("barrier")

    def spawn(self, fn: Callable, *args: Any, **kwargs: Any) -> None:
        return xmp.spawn(
            fn, args=(self._world_size,), nprocs=self._world_size, start_method="fork"
        )

    def setup_process(self, rank: int = -1, world_size: int = 1):
        """Initialize DDP variables and processes.

        Args:
            rank: process rank. Default is `-1`.
            world_size: number of devices in netwok to expect for train. Default is `1`.
        """
        self._rank = rank
        self._world_size = world_size
        self._device = xm.xla_device()

    def sync_tensor(self, tensor: torch.Tensor, mode: str) -> torch.Tensor:
        """Syncs ``tensor`` over ``world_size`` in distributed mode.

        Args:
            tensor: tensor to sync across the processes.
            mode: tensor synchronization type,
                should be one of 'sum' or 'mean'.
                Default is 'mean'.

        Returns:
            torch.Tensor with synchronized values.
        """
        # return tensor
        if mode not in {"sum", "mean"}:
            raise ValueError(f"Unknown sync_type '{mode}'")
        if mode == "sum":
            return xm.all_reduce("sum", tensor)
        elif mode == "mean":
            return xm.all_reduce("sum", tensor, scale=1.0 / self.world_size)

    def sync_metrics(self, metrics: Dict) -> Dict:
        metrics = {
            k: xm.mesh_reduce(k, v.item() if isinstance(v, torch.Tensor) else v, np.mean)
            for k, v in metrics.items()
        }
        return metrics

    def optimizer_step(self, loss, model, optimizer) -> None:
        """Abstraction over ``optimizer.step()`` step."""
        xm.optimizer_step(optimizer)

    def autocast_loader(self, loader):
        return ParallelLoader(loader, [self.device]).per_device_loader(self.device)


__all__ = ["XLAEngine", "DistributedXLAEngine"]
