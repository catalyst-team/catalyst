import os

import torch
import torch.distributed as dist
import torch.nn as nn

from catalyst.engines.device import DeviceEngine


class DistributedDataParallelEngine(DeviceEngine):
    def __init__(
        self,
        rank: int,
        world_size: int,
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

    def __repr__(self):
        return (
            f"DistributedDeviceEngine(address={self.address},"
            f"port={self.port},backend='{self.backend}',"
            f"rank={self.device},world_size={self.world_size})"
        )

    def setup_experiment(self):
        """Initialize DDP variables and processes."""
        os.environ["MASTER_ADDR"] = str(self.address)
        os.environ["MASTER_PORT"] = str(self.port)
        dist.init_process_group(self.backend, rank=self.device, world_size=self.world_size)

    def cleanup(self):
        dist.destroy_process_group()

    def sync_metric(self):
        pass

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
