from typing import Any, Dict, Mapping, Union
import os

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel

from catalyst.core.engine import IEngine
from catalyst.engines.functional import mean_reduce, sum_reduce
from catalyst.typing import RunnerCriterion, RunnerModel, RunnerOptimizer, RunnerScheduler
from catalyst.utils.distributed import get_nn_from_ddp_module
from catalyst.utils.misc import maybe_recursive_call


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
        checkpoint = kwargs

        if isinstance(model, dict):
            for key, value in model.items():
                model_module = get_nn_from_ddp_module(value)
                checkpoint[f"model_{key}_state_dict"] = maybe_recursive_call(
                    model_module, "state_dict"
                )
        else:
            model_module = get_nn_from_ddp_module(model)
            checkpoint["model_state_dict"] = maybe_recursive_call(model_module, "state_dict")

        for dict2save, name2save in zip(
            [criterion, optimizer, scheduler], ["criterion", "optimizer", "scheduler"],
        ):
            if dict2save is None:
                continue
            if isinstance(dict2save, dict):
                for key, value in dict2save.items():
                    if value is not None:
                        state_dict2save = name2save + "_" + str(key)
                        # checkpoint[name2save_] = value
                        state_dict2save = state_dict2save + "_state_dict"
                        checkpoint[state_dict2save] = value.state_dict()
            else:
                # checkpoint[name2save] = dict2save
                name2save = name2save + "_state_dict"
                checkpoint[name2save] = dict2save.state_dict()
        return checkpoint

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

        if model is not None:
            model = get_nn_from_ddp_module(model)
            maybe_recursive_call(
                model, "load_state_dict", recursive_args=checkpoint["model_state_dict"],
            )

        for dict2load, name2load in zip(
            [criterion, optimizer, scheduler], ["criterion", "optimizer", "scheduler"],
        ):
            if dict2load is None:
                continue

            if isinstance(dict2load, dict):
                for key, value in dict2load.items():
                    if value is not None:
                        state_dict2load = f"{name2load}_{key}_state_dict"
                        value.load_state_dict(checkpoint[state_dict2load])
            else:
                name2load = f"{name2load}_state_dict"
                dict2load.load_state_dict(checkpoint[name2load])

    def save_checkpoint(self, checkpoint: Mapping[str, Any], path: str):
        """@TODO: docs."""
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str):
        """@TODO: docs."""
        checkpoint = torch.load(path, map_location=lambda storage, loc: storage)
        return checkpoint


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
