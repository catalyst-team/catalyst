from typing import Any, Dict, Mapping, Union
import os

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel

from catalyst.core.engine import IEngine
from catalyst.typing import RunnerCriterion, RunnerModel, RunnerOptimizer, RunnerScheduler
from catalyst.utils.distributed import mean_reduce, sum_reduce
from catalyst.utils.torch import (
    any2device,
    load_checkpoint,
    pack_checkpoint,
    save_checkpoint,
    unpack_checkpoint,
)


class DeviceEngine(IEngine):
    """Single training device engine."""

    def __init__(self, device: str = None):
        """
        Args:
            device (str, optional): use device, default is `"cpu"`.
        """
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

    def __repr__(self) -> str:  # noqa: D105
        return f"{self.__class__.__name__}(device='{self.device}')"

    @property
    def rank(self) -> int:
        """@TODO: docs."""
        return -1

    @property
    def world_size(self) -> int:
        """@TODO: docs."""
        return 1

    def sync_device(
        self, tensor_or_module: Union[dict, list, tuple, torch.Tensor, nn.Module]
    ) -> Any:
        """@TODO: docs."""
        return any2device(tensor_or_module, device=self.device)

    def sync_tensor(self, tensor: Any, *args, **kwargs) -> Any:
        """@TODO: docs."""
        return tensor

    def init_components(
        self, model_fn=None, criterion_fn=None, optimizer_fn=None, scheduler_fn=None,
    ):
        """@TODO: docs."""
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

    def deinit_components(self):
        """@TODO: docs."""
        # remove backend
        pass

    def zero_grad(self, loss, model, optimizer) -> None:
        """@TODO: docs."""
        model.zero_grad()

    def backward_loss(self, loss, model, optimizer) -> None:
        """@TODO: docs."""
        loss.backward()

    def optimizer_step(self, loss, model, optimizer) -> None:
        """@TODO: docs."""
        optimizer.step()

    def pack_checkpoint(
        self,
        model: RunnerModel = None,
        criterion: RunnerCriterion = None,
        optimizer: RunnerOptimizer = None,
        scheduler: RunnerScheduler = None,
        **kwargs,
    ) -> Dict:
        """@TODO: docs."""
        return pack_checkpoint(
            model=model, criterion=criterion, optimizer=optimizer, scheduler=scheduler, **kwargs
        )

    def unpack_checkpoint(
        self,
        checkpoint: Dict,
        model: RunnerModel = None,
        criterion: RunnerCriterion = None,
        optimizer: RunnerOptimizer = None,
        scheduler: RunnerScheduler = None,
        **kwargs,
    ) -> None:
        """@TODO: docs."""
        return unpack_checkpoint(
            checkpoint=checkpoint,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
        )

    def save_checkpoint(self, checkpoint: Mapping[str, Any], path: str):
        """@TODO: docs."""
        return save_checkpoint(checkpoint=checkpoint, path=path)

    def load_checkpoint(self, path: str):
        """@TODO: docs."""
        return load_checkpoint(path=path)


class DataParallelEngine(DeviceEngine):
    """@TODO: docs."""

    def __init__(self):
        """@TODO: docs."""
        super().__init__(f"cuda:{torch.cuda.current_device()}")
        self.device_count = torch.cuda.device_count()

    def __repr__(self) -> str:  # noqa: D105
        return f"{self.__class__.__name__}(device_count={self.device_count})"

    def init_components(
        self, model_fn=None, criterion_fn=None, optimizer_fn=None, scheduler_fn=None,
    ):
        """@TODO: docs."""
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


class DistributedDataParallelEngine(DeviceEngine):
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
            address: process address to use (required for PyTorch backend),
                default is `"localhost"`.
            port: process port to listen (required for PyTorch backend), default is `"12345"`.
            backend: multiprocessing backend to use, default is `"nccl"`.
            world_size: number of processes.
        """
        super().__init__()
        self.address = address
        self.port = port
        self.backend = backend
        self._rank = 0
        self._world_size = world_size or torch.cuda.device_count()
        self.device = None

    def __repr__(self):  # noqa: D105
        return (
            f"{self.__class__.__name__}(address={self.address}, "
            f"port={self.port}, backend='{self.backend}',"
            f"rank={self._rank}, world_size={self._world_size})"
        )

    @property
    def rank(self) -> int:
        """@TODO: docs."""
        return self._rank

    @property
    def world_size(self) -> int:
        """@TODO: docs."""
        return self._world_size

    @property
    def is_master_process(self) -> bool:
        """@TODO: docs."""
        return self._rank == 0

    @property
    def is_worker_process(self) -> bool:
        """@TODO: docs."""
        return self._rank > 0

    def setup_process(self, rank: int = -1, world_size: int = 1):
        """Initialize DDP variables and processes."""
        self._rank = rank
        self._world_size = world_size
        os.environ["MASTER_ADDR"] = str(self.address)
        os.environ["MASTER_PORT"] = str(self.port)
        dist.init_process_group(self.backend, rank=self.rank, world_size=self.world_size)
        torch.cuda.set_device(int(self._rank))
        self.device = f"cuda:{int(self._rank)}"

    def cleanup_process(self):
        """Clean DDP variables and processes."""
        dist.destroy_process_group()

    # @TODO: add all_gather
    def sync_tensor(self, tensor: torch.Tensor, mode: str):
        """Synchronize tensor.

        Args:
            tensor: tensor to sync across the processes.
            mode: tensor synchronization type,
                should be one of 'sum' or 'mean'.
                Default is 'mean'.

        Returns:
            torch.Tensor with synchronized values.

        Raises:
            ValueError: if mode is out of ``sum`` or ``mean``
        """
        if mode not in {"sum", "mean"}:
            raise ValueError(f"Unknown sync_type '{mode}'")
        if mode == "sum":
            return sum_reduce(tensor)
        else:
            return mean_reduce(tensor, self.world_size)

    def init_components(
        self, model_fn=None, criterion_fn=None, optimizer_fn=None, scheduler_fn=None,
    ):
        """@TODO: docs."""
        model = model_fn()
        model = self.sync_device(model)
        # NOTE: do not forget to wrap a model in DDP
        model = DistributedDataParallel(
            model, device_ids=[self.device], find_unused_parameters=True
        )
        # criterion
        criterion = criterion_fn()
        criterion = self.sync_device(criterion)
        # optimizer
        optimizer = optimizer_fn()
        optimizer = self.sync_device(optimizer)
        # scheduler
        scheduler = scheduler_fn()
        scheduler = self.sync_device(scheduler)
        dist.barrier()
        return model, criterion, optimizer, scheduler

    def deinit_components(self):
        """@TODO: docs."""
        self.cleanup_process()

    def zero_grad(self, loss, model, optimizer) -> None:
        """@TODO: docs."""
        model.zero_grad()

    def backward_loss(self, loss, model, optimizer) -> None:
        """@TODO: docs."""
        loss.backward()

    def optimizer_step(self, loss, model, optimizer) -> None:
        """@TODO: docs."""
        optimizer.step()
        dist.barrier()


__all__ = ["DeviceEngine", "DataParallelEngine", "DistributedDataParallelEngine"]
