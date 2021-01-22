from typing import Dict
import os

import torch
import torch.distributed as dist
import torch.nn as nn

from catalyst.engines.device import DeviceEngine
from catalyst.engines.functional import mean_reduce, sum_reduce


class DistributedDataParallelEngine(DeviceEngine):
    def __init__(
        self,
        rank: int = 0,
        world_size: int = 1,
        address: str = "localhost",
        port: str = "12345",
        backend: str = "nccl",
    ):
        """
        Args:
            rank: process rank
            world_size: total number of processes in experiment
        """
        super().__init__(rank)
        self._world_size = world_size
        self.address = address
        self.port = port
        self.backend = backend

    @property
    def rank(self) -> int:
        return self.device

    @property
    def world_size(self) -> int:
        return self._world_size

    def __repr__(self):  # noqa: D105
        return (
            f"DistributedDataParallelEngine(address={self.address},"
            f"port={self.port},backend='{self.backend}',"
            f"rank={self.device},world_size={self._world_size})"
        )

    def init_process(self):
        """Initialize DDP variables and processes."""
        os.environ["MASTER_ADDR"] = str(self.address)
        os.environ["MASTER_PORT"] = str(self.port)
        dist.init_process_group(self.backend, rank=self.device, world_size=self.world_size)

    def init_components(
        self, model_fn=None, criterion_fn=None, optimizer_fn=None, scheduler_fn=None,
    ):
        self.init_process()
        return super().init_components(model_fn, criterion_fn, optimizer_fn, scheduler_fn)

    def cleanup_process(self):
        """Clean DDP variables and processes."""
        dist.destroy_process_group()

    def sync_metric(self, tensor, sync_type="mean"):
        """Synchronize tensor.

        Args:
            tensor (torch.Tensor): tensor to sync across the processes.
            sync_type (str): tensor synchronization type,
                should be one of 'sum' or 'mean'.
                Default is 'mean'.

        Returns:
            torch.Tensor with synchronized values.
        """
        if sync_type not in {"sum", "mean"}:
            raise ValueError(f"Unknown sync_type '{sync_type}'")
        if sync_type == "sum":
            return sum_reduce(tensor)
        else:
            return mean_reduce(tensor, self.world_size)

    def pack_checkpoint(
        self, model=None, criterion=None, optimizer=None, scheduler=None, **kwargs,
    ) -> Dict:
        # unwrap model
        _model = model.module if isinstance(model, nn.parallel.DistributedDataParallel) else model
        return super().pack_checkpoint(_model, criterion, optimizer, scheduler, **kwargs)
