from typing import Any, Dict, Union
import copy
import os

import torch
import torch.distributed as dist

from catalyst.engines.torch import DeviceEngine
from catalyst.settings import SETTINGS
from catalyst.utils.distributed import mean_reduce, sum_reduce

if SETTINGS.deepspeed_required:
    import deepspeed


# @TODO: create distributed abstraction
class DistributedDataParallelDeepSpeedEngine(DeviceEngine):
    """Distributed DeepSpeed MultiGPU training device engine.

    Args:
        address: address to use for backend.
        port: port to use for backend.
        process_group_kwargs: parameters for `torch.distributed.init_process_group`.
            More info here:
            https://pytorch.org/docs/stable/distributed.html#torch.distributed.init_process_group
        deepspeed_kwargs: parameters for `deepspeed.initialize`
    """

    def __init__(
        self,
        address: str = None,
        port: Union[str, int] = None,
        train_batch_size: int = 256,
        process_group_kwargs: Dict[str, Any] = None,
        deepspeed_kwargs: Dict[str, Any] = None,
    ):
        """Init."""
        super().__init__()
        self.address = address or "localhost"
        self.port = port or 12345
        self.train_batch_size = train_batch_size
        self._rank = 0
        self.device = None

        if process_group_kwargs is None:
            process_group_kwargs = {}
        self.process_group_kwargs = copy.deepcopy(process_group_kwargs)

        self._world_size = (
            self.process_group_kwargs.get("world_size", None) or torch.cuda.device_count()
        )
        self.deepspeed_kwargs = deepspeed_kwargs or {}

    def __repr__(self):  # noqa: D105
        return (
            f"{self.__class__.__name__}(address={self.address}, "
            f"port={self.port}, "
            f"process_group_kwargs={self.process_group_kwargs}, "
            f"deepspeed_kwargs={self.deepspeed_kwargs})"
        )

    @property
    def rank(self) -> int:
        """Process rank for distributed training."""
        return self._rank

    @property
    def world_size(self) -> int:
        """Process world size  for distributed training."""
        return self._world_size

    @property
    def is_master_process(self) -> bool:
        """Checks if a process is master process.
        Should be implemented only for DDP setup in other cases should always return True.

        Returns:
            `True` if current process is a master process, otherwise `False`.
        """
        return self._rank == 0

    @property
    def is_worker_process(self) -> bool:
        """Checks if a process is worker process.
        Should be implemented only for DDP setup in other cases should always return False.

        Returns:
            `True` if current process is a worker process, otherwise `False`.
        """
        return self._rank > 0

    def setup_process(self, rank: int = -1, world_size: int = 1):
        """Initialize DDP variables and processes.

        Args:
            rank: process rank. Default is `-1`.
            world_size: number of devices in netwok to expect for train.
                Default is `1`.
        """
        self._rank = rank
        self._world_size = world_size

        # self.process_group_kwargs["rank"] = rank
        # self.process_group_kwargs["world_size"] = world_size
        os.environ["RANK"] = str(rank)
        os.environ["LOCAL_RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["MASTER_ADDR"] = str(self.address)
        os.environ["MASTER_PORT"] = str(self.port)

        deepspeed.init_distributed(**self.process_group_kwargs)

        torch.cuda.set_device(int(self._rank))
        self.device = f"cuda:{int(self._rank)}"

    def cleanup_process(self):
        """Clean DDP variables and processes."""
        dist.destroy_process_group()

    # @TODO: add all_gather
    def sync_tensor(self, tensor: torch.Tensor, mode: str):
        """Syncs ``tensor`` over ``world_size`` in distributed mode.

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
        """Inits the runs components."""
        model = model_fn()
        model = self.sync_device(model)

        criterion = criterion_fn()
        criterion = self.sync_device(criterion)

        optimizer = optimizer_fn()
        optimizer = self.sync_device(optimizer)

        model, optimizer, _, _ = deepspeed.initialize(
            model=model,
            optimizer=optimizer,
            # @TODO: not sure about this deepspeed feature
            config={"train_batch_size": self.train_batch_size},
            **self.deepspeed_kwargs,
        )

        scheduler = scheduler_fn()
        scheduler = self.sync_device(scheduler)
        return model, criterion, optimizer, scheduler

    def deinit_components(self):
        """Deinits the runs components."""
        dist.barrier()
        self.cleanup_process()

    def zero_grad(self, loss, model, optimizer) -> None:
        """Abstraction over ``model.zero_grad()`` step."""
        model.zero_grad()

    def backward_loss(self, loss, model, optimizer) -> None:
        """Abstraction over ``loss.backward()`` step."""
        model.backward(loss)

    def optimizer_step(self, loss, model, optimizer) -> None:
        """Abstraction over ``optimizer.step()`` step."""
        model.step()


__all__ = ["DistributedDataParallelDeepSpeedEngine"]
