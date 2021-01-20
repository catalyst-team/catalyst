import os

import torch
import torch.distributed as dist
import torch.nn as nn

from catalyst.engines.device import DeviceEngine
from catalyst.engines.functional import sum_reduce, mean_reduce


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
        self.world_size = world_size
        self.address = address
        self.port = port
        self.backend = backend

    def __repr__(self):  # noqa: D105
        return (
            f"DistributedDataParallelEngine(address={self.address},"
            f"port={self.port},backend='{self.backend}',"
            f"rank={self.device},world_size={self.world_size})"
        )

    def init_process(self):
        """Initialize DDP variables and processes."""
        os.environ["MASTER_ADDR"] = str(self.address)
        os.environ["MASTER_PORT"] = str(self.port)
        dist.init_process_group(self.backend, rank=self.device, world_size=self.world_size)

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

    # TODO: maybe handle unpacking DataParallel to simple nn.Module
    def load_checkpoint(
        self,
        file: str,
        model: nn.DataParallel,
        optimizer: nn.Module = None,
        criterion=None,
        scheduler=None,
    ):
        content = torch.load(file)

        if "model_state_dict" in content:
            model.module.load_state_dict(content["model_state_dict"])

        if "optimizer_state_dict" in content and optimizer is not None:
            optimizer.load_state_dict(content["optimizer_state_dict"])

        if "criterion_state_dict" in content and criterion is not None:
            criterion.load_state_dict(content["criterion_state_dict"])

        if "scheduler_state_dict" in content and scheduler is not None:
            scheduler.load_state_dict(content["scheduler_state_dict"])
