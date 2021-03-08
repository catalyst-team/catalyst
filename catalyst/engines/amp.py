# flake8: noqa
# TODO: works only with latest pytorch (1.7.1) - fix for older versions
from typing import Any, Dict, Mapping

import torch
import torch.cuda.amp as amp
import torch.nn as nn
from torch.nn.parallel import DataParallel

from catalyst.engines.device import DeviceEngine
from catalyst.engines.distributed import DistributedDataParallelEngine


class AMPEngine(DeviceEngine):
    """@TODO: docs."""

    def __init__(self, device: str = "cuda"):
        """
        Args:
            device (str): use device, default is `"cpu"`.
        """
        super().__init__(device)
        self.scaler = amp.GradScaler()

    def __repr__(self) -> str:  # noqa: D105
        return f"{self.__class__.__name__}(device='{self.device}')"

    def backward_loss(self, loss, model, optimizer) -> None:
        """@TODO: docs."""
        self.scaler.scale(loss).backward()

    def optimizer_step(self, loss, model, optimizer) -> None:
        """@TODO: docs."""
        self.scaler.step(optimizer)
        self.scaler.update()

    def init_components(
        self, model_fn=None, criterion_fn=None, optimizer_fn=None, scheduler_fn=None,
    ):
        """@TODO: docs."""
        # TODO: how could we do better?)
        # model
        model = model_fn()
        model = self.sync_device(model)
        # criterion
        criterion = criterion_fn()
        criterion = self.sync_device(criterion)
        # optimizer
        optimizer = optimizer_fn()
        optimizer = self.sync_device(optimizer)
        # scheduler
        scheduler = scheduler_fn()
        scheduler = self.sync_device(scheduler)
        return model, criterion, optimizer, scheduler

    # TODO: should be used with forward method? (similar to criterion)
    def autocast(self):
        return amp.autocast()


class DataParallelAMPEngine(AMPEngine):
    def __init__(self):
        super().__init__(f"cuda:{torch.cuda.current_device()}")
        self.device_count = torch.cuda.device_count()

    def __repr__(self) -> str:  # noqa: D105
        return f"{self.__class__.__name__}(device='{self.device}')"

    def init_components(
        self, model_fn=None, criterion_fn=None, optimizer_fn=None, scheduler_fn=None,
    ):
        model = model_fn()
        model = self.sync_device(model)
        model = DataParallel(model)

        # criterion
        criterion = criterion_fn()
        criterion = self.sync_device(criterion)
        # optimizer
        optimizer = optimizer_fn()
        optimizer = self.sync_device(optimizer)
        # scheduler
        scheduler = scheduler_fn()
        scheduler = self.sync_device(scheduler)

        return model, criterion, optimizer, scheduler

    def pack_checkpoint(
        self, model=None, criterion=None, optimizer=None, scheduler=None, **kwargs,
    ) -> Dict:
        # unwrap model
        _model = model.module if isinstance(model, nn.DataParallel) else model
        return super().pack_checkpoint(_model, criterion, optimizer, scheduler, **kwargs)

    def save_checkpoint(self, checkpoint: Mapping[str, Any], path: str):
        # TODO: method for unpacking torch.nn.DataParallel
        torch.save(checkpoint, path)


# TODO: move this class to a engines/distributed.py ??
class DistributedDataParallelAMPEngine(DistributedDataParallelEngine):
    """@TODO: docs."""

    def __init__(
        self,
        address: str = "localhost",
        port: str = "12345",
        backend: str = "nccl",
        world_size: int = None,
    ):
        """
        Args:
            address (str): process address to use (required for PyTorch backend),
                default is `"localhost"`.
            port (str): process port to listen (required for PyTorch backend),
                default is `"12345"`.
            backend (str): multiprocessing backend to use,
                default is `"nccl"`.
            world_size (int): number of processes.
        """
        super().__init__(address, port, backend, world_size)
        self.scaler = amp.GradScaler()

    def __repr__(self):  # noqa: D105
        return (
            f"{self.__class__.__name__}(address={self.address}, "
            f"port={self.port}, backend='{self.backend}',"
            f"rank={self._rank}, world_size={self._world_size})"
        )

    def backward_loss(self, loss, model, optimizer) -> None:
        """@TODO: docs."""
        self.scaler.scale(loss).backward()

    def optimizer_step(self, loss, model, optimizer) -> None:
        """@TODO: docs."""
        self.scaler.step(optimizer)
        self.scaler.update()

    # TODO: should be used with forward method? (similar to criterion)
    def autocast(self):
        """@TODO: docs."""
        return amp.autocast()


__all__ = ["AMPEngine", "DistributedDataParallelAMPEngine"]
