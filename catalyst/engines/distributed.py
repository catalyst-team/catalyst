# flake8: noqa
from typing import Any, Dict, Mapping, Union
import os

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel

from catalyst.core.engine import IEngine
from catalyst.engines.functional import mean_reduce, sum_reduce


class DistributedDataParallelEngine(IEngine):
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
        return self._rank

    @property
    def world_size(self) -> int:
        return self._world_size

    @property
    def is_master_process(self) -> bool:
        return self._rank == 0

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

    def sync_device(
        self, tensor_or_module: Union[dict, list, tuple, torch.Tensor, nn.Module]
    ) -> Any:
        if isinstance(tensor_or_module, dict):
            return {key: self.sync_device(value) for key, value in tensor_or_module.items()}
        elif isinstance(tensor_or_module, (list, tuple)):
            return type(tensor_or_module)(self.sync_device(elem) for elem in tensor_or_module)
        elif torch.is_tensor(tensor_or_module):
            return tensor_or_module.to(self.device, non_blocking=True)
        elif (
            isinstance(tensor_or_module, (np.ndarray, np.void))
            and tensor_or_module.dtype.fields is not None
        ):
            return {
                k: self.sync_device(tensor_or_module[k])
                for k in tensor_or_module.dtype.fields.keys()
            }
        elif isinstance(tensor_or_module, np.ndarray):
            return torch.tensor(tensor_or_module, device=self.device)
        elif isinstance(tensor_or_module, nn.Module):
            return tensor_or_module.to(self.device)
        # elif hasattr(tensor_or_module, "to"):
        #     return tensor_or_module.to(self.device)
        return tensor_or_module

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
        # self.setup_process()

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
        self.cleanup_process()

    def zero_grad(self, loss, model, optimizer) -> None:
        model.zero_grad()

    def backward_loss(self, loss, model, optimizer) -> None:
        loss.backward()

    def optimizer_step(self, loss, model, optimizer) -> None:
        optimizer.step()
        dist.barrier()

    def pack_checkpoint(
        self, model=None, criterion=None, optimizer=None, scheduler=None, **kwargs,
    ) -> Dict:
        _model = model.module if isinstance(model, DistributedDataParallel) else model
        return {
            "model": _model,
            "criterion": criterion,
            "optimizer": optimizer,
            "scheduler": scheduler,
            **kwargs,
        }

    def unpack_checkpoint(
        self,
        checkpoint: Dict,
        model=None,
        criterion=None,
        optimizer=None,
        scheduler=None,
        **kwargs,
    ) -> None:

        if "model_state_dict" in checkpoint:
            if isinstance(model, nn.DataParallel):
                model.module.load_state_dict(checkpoint["model_state_dict"])
            elif isinstance(model, nn.Module):
                model.load_state_dict(checkpoint["model_state_dict"])

        if "optimizer_state_dict" in checkpoint and optimizer is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if "criterion_state_dict" in checkpoint and criterion is not None:
            criterion.load_state_dict(checkpoint["criterion_state_dict"])

        if "scheduler_state_dict" in checkpoint and scheduler is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    def save_checkpoint(self, checkpoint: Mapping[str, Any], path: str):
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str):
        return torch.load(path)


__all__ = ["DistributedDataParallelEngine"]
