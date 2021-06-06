from typing import Any, Dict, Union
import os

import torch

from catalyst.engines.torch import DistributedDataParallelEngine
from catalyst.settings import SETTINGS

if SETTINGS.deepspeed_required:
    import deepspeed


class DistributedDataParallelDeepSpeedEngine(DistributedDataParallelEngine):
    """Distributed DeepSpeed MultiGPU training device engine.

    Args:
        address: address to use for backend.
        port: port to use for backend.
        ddp_kwargs: parameters for `apex.parallel.DistributedDataParallel`.
            More info here:
            https://nvidia.github.io/apex/parallel.html#apex.parallel.DistributedDataParallel
        process_group_kwargs: parameters for `torch.distributed.init_process_group`.
            More info here:
            https://pytorch.org/docs/stable/distributed.html#torch.distributed.init_process_group
        deepspeed_kwargs: parameters for `deepspeed.initialize`
    """

    def __init__(
        self,
        address: str = None,
        port: Union[str, int] = None,
        ddp_kwargs: Dict[str, Any] = None,
        process_group_kwargs: Dict[str, Any] = None,
        deepspeed_kwargs: Dict[str, Any] = None,
    ):
        """Init."""
        super().__init__(
            address=address, port=port, ddp_kwargs=None, process_group_kwargs=process_group_kwargs
        )
        self.ddp_kwargs = ddp_kwargs or {}
        self.deepspeed_kwargs = deepspeed_kwargs or {}

    def __repr__(self):  # noqa: D105
        return (
            f"{self.__class__.__name__}(address={self.address}, "
            f"port={self.port}, "
            f"ddp_kwargs={self.ddp_kwargs}, "
            f"process_group_kwargs={self.process_group_kwargs}, "
            f"deepspeed_kwargs={self.deepspeed_kwargs})"
        )

    def setup_process(self, rank: int = -1, world_size: int = 1):
        """Initialize DDP variables and processes.

        Args:
            rank: process rank. Default is `-1`.
            world_size: number of devices in netwok to expect for train.
                Default is `1`.
        """
        self._rank = rank
        self._world_size = world_size

        self.process_group_kwargs["rank"] = rank
        self.process_group_kwargs["world_size"] = world_size
        os.environ["MASTER_ADDR"] = str(self.address)
        os.environ["MASTER_PORT"] = str(self.port)

        deepspeed.init_distributed(**self.process_group_kwargs)

        torch.cuda.set_device(int(self._rank))
        self.device = f"cuda:{int(self._rank)}"

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
            model=model, optimizer=optimizer, **self.deepspeed_kwargs
        )

        scheduler = scheduler_fn()
        scheduler = self.sync_device(scheduler)
        return model, criterion, optimizer, scheduler

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
