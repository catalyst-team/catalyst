from typing import Any, Dict, Union

import torch
import torch.cuda.amp as amp

from catalyst.engines.torch import DeviceEngine, DistributedDataParallelEngine
from catalyst.settings import SETTINGS

if SETTINGS.fairscale_required:
    from fairscale.nn import Pipe
    from fairscale.nn.data_parallel import (
        FullyShardedDataParallel as FSDP,
        ShardedDataParallel as ShardedDDP,
    )
    from fairscale.optim.grad_scaler import ShardedGradScaler


class PipelineParallelFairScaleEngine(DeviceEngine):
    """FairScale multi-gpu training device engine.

    Args:
        pipe_kwargs: parameters for `fairscale.nn.Pipe`

            Docs for `fairscale.nn.Pipe`:
            https://fairscale.readthedocs.io/en/latest/api/nn/pipe.html

    """

    def __init__(self, pipe_kwargs: Dict[str, Any] = None):
        """Init."""
        super().__init__(f"cuda:{torch.cuda.current_device()}")
        self.device_count = torch.cuda.device_count()
        self.pipe_kwargs = pipe_kwargs

    def __repr__(self) -> str:  # noqa: D105
        return f"{self.__class__.__name__}(device_count={self.device_count})"

    def init_components(
        self, model_fn=None, criterion_fn=None, optimizer_fn=None, scheduler_fn=None,
    ):
        """Inits the runs components."""
        model = model_fn()
        model = self.sync_device(model)

        # criterion
        criterion = criterion_fn()
        criterion = self.sync_device(criterion)

        # optimizer
        optimizer = optimizer_fn()
        optimizer = self.sync_device(optimizer)

        model = Pipe(model, **self.pipe_kwargs)

        # scheduler
        scheduler = scheduler_fn()
        scheduler = self.sync_device(scheduler)
        return model, criterion, optimizer, scheduler


class SharedDataParallelFairScaleEngine(DistributedDataParallelEngine):
    """Distributed FairScale MultiGPU training device engine.

    Args:
        address: address to use for backend.
        port: port to use for backend.
        ddp_kwargs: parameters for `fairscale.nn.data_parallel.ShardedDataParallel`.
            More info here:
            https://fairscale.readthedocs.io/en/latest/api/nn/sharded_ddp.html
        process_group_kwargs: parameters for `torch.distributed.init_process_group`.
            More info here:
            https://pytorch.org/docs/stable/distributed.html#torch.distributed.init_process_group
    """

    def __init__(
        self,
        address: str = None,
        port: Union[str, int] = None,
        ddp_kwargs: Dict[str, Any] = None,
        process_group_kwargs: Dict[str, Any] = None,
    ):
        """Init."""
        super().__init__(
            address=address, port=port, ddp_kwargs=None, process_group_kwargs=process_group_kwargs
        )
        self.ddp_kwargs = ddp_kwargs or {}

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

        model = ShardedDDP(model, optimizer, **self.ddp_kwargs)

        scheduler = scheduler_fn()
        scheduler = self.sync_device(scheduler)
        return model, criterion, optimizer, scheduler


class SharedDataParallelFairScaleAMPEngine(SharedDataParallelFairScaleEngine):
    """Distributed FairScale MultiGPU training device engine.

    Args:
        address: address to use for backend.
        port: port to use for backend.
        ddp_kwargs: parameters for `fairscale.nn.data_parallel.ShardedDataParallel`.
        process_group_kwargs: parameters for `torch.distributed.init_process_group`.
            More info here:
            https://pytorch.org/docs/stable/distributed.html#torch.distributed.init_process_group
        scaler_kwargs: parameters for `fairscale.optim.grad_scaler.ShardedGradScaler`.
            Possible parameters:
            https://fairscale.readthedocs.io/en/latest/api/index.html
    """

    def __init__(
        self,
        address: str = None,
        port: Union[str, int] = None,
        ddp_kwargs: Dict[str, Any] = None,
        process_group_kwargs: Dict[str, Any] = None,
        scaler_kwargs: Dict[str, Any] = None,
    ):
        """Init."""
        super().__init__(
            address=address,
            port=port,
            ddp_kwargs=ddp_kwargs,
            process_group_kwargs=process_group_kwargs,
        )
        if scaler_kwargs is None:
            scaler_kwargs = {}
        self.scaler_kwargs = scaler_kwargs
        self.scaler = ShardedGradScaler(**self.scaler_kwargs)

    def zero_grad(self, loss, model, optimizer) -> None:
        """Abstraction over ``model.zero_grad()`` step."""
        optimizer.zero_grad()

    def backward_loss(self, loss, model, optimizer) -> None:
        """Abstraction over ``loss.backward()`` step."""
        self.scaler.scale(loss).backward()

    def optimizer_step(self, loss, model, optimizer) -> None:
        """Abstraction over ``optimizer.step()`` step."""
        self.scaler.step(optimizer)
        self.scaler.update()

    def autocast(self):
        """AMP context"""
        return amp.autocast()


class FullySharedDataParallelFairScaleEngine(DistributedDataParallelEngine):
    """Distributed FairScale MultiGPU training device engine.

    Args:
        address: address to use for backend.
        port: port to use for backend.
        ddp_kwargs: parameters for `fairscale.nn.data_parallel.FullyShardedDataParallel`.
        process_group_kwargs: parameters for `torch.distributed.init_process_group`.
            More info here:
            https://pytorch.org/docs/stable/distributed.html#torch.distributed.init_process_group
    """

    def __init__(
        self,
        address: str = None,
        port: Union[str, int] = None,
        ddp_kwargs: Dict[str, Any] = None,
        process_group_kwargs: Dict[str, Any] = None,
    ):
        """Init."""
        super().__init__(
            address=address, port=port, ddp_kwargs=None, process_group_kwargs=process_group_kwargs
        )
        self.ddp_kwargs = ddp_kwargs or {}

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

        model = FSDP(model, **self.ddp_kwargs)

        scheduler = scheduler_fn()
        scheduler = self.sync_device(scheduler)
        return model, criterion, optimizer, scheduler


__all__ = [
    "PipelineParallelFairScaleEngine",
    "SharedDataParallelFairScaleEngine",
    "SharedDataParallelFairScaleAMPEngine",
    "FullySharedDataParallelFairScaleEngine",
]
