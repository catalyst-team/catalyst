import os
import torch.nn as nn
import torch.distributed as dist

from catalyst.engines.device import DeviceEngine


class DistributedDeviceEngine(DeviceEngine):
    def __init__(
        self,
        address: str = "localhost",
        port: str = "12345",
        backend: str = "nccl",
    ):
        self.device = "ddp"
        self.address = address
        self.port = port
        self.backend = backend

    def __repr__(self):
        return (
            f"DistributedDeviceEngine(address={self.address},"
            f"port={self.port},backend='{self.backend}')"
        )

    def setup_experiment(self, rank: int, world_size: int):
        """
        Args:
            rank: process rank
            world_size: total number of processes in experiment
        """
        os.environ["MASTER_ADDR"] = str(self.address)
        os.environ["MASTER_PORT"] = str(self.port)
        dist.init_process_group(self.backend, rank=rank, world_size=world_size)

    def cleanup(self):
        dist.destroy_process_group()

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

