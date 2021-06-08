from typing import Any, Dict, Union
import os

import torch
import torch.cuda.amp as amp
import torch.distributed as dist

from catalyst.engines.torch import DeviceEngine, DistributedDataParallelEngine
from catalyst.settings import SETTINGS
from catalyst.typing import RunnerCriterion, RunnerModel, RunnerOptimizer, RunnerScheduler

if SETTINGS.fairscale_required:
    from fairscale.nn import Pipe
    from fairscale.nn.data_parallel import (
        FullyShardedDataParallel as FSDP,
        ShardedDataParallel as ShardedDDP,
    )
    from fairscale.optim import OSS
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
        # model = self.sync_device(model)

        model = Pipe(model, **self.pipe_kwargs)

        # criterion
        criterion = criterion_fn()
        # criterion = self.sync_device(criterion)

        # optimizer
        optimizer = optimizer_fn()
        # optimizer = self.sync_device(optimizer)

        # scheduler
        scheduler = scheduler_fn()
        # scheduler = self.sync_device(scheduler)
        return model, criterion, optimizer, scheduler

    def zero_grad(self, loss, model, optimizer) -> None:
        """Abstraction over ``model.zero_grad()`` step."""
        optimizer.zero_grad()

    def backward_loss(self, loss, model, optimizer) -> None:
        """Abstraction over ``loss.backward()`` step."""
        loss.backward()

    def optimizer_step(self, loss, model, optimizer) -> None:
        """Abstraction over ``optimizer.step()`` step."""
        optimizer.step()


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

        dist.init_process_group(**self.process_group_kwargs)

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

        optimizer = OSS(model.parameters(), optim=optimizer.__class__, **optimizer.defaults)
        model = ShardedDDP(model, optimizer, **self.ddp_kwargs)

        scheduler = scheduler_fn()
        scheduler = self.sync_device(scheduler)
        return model, criterion, optimizer, scheduler

    def pack_checkpoint(
        self,
        model: RunnerModel = None,
        criterion: RunnerCriterion = None,
        optimizer: RunnerOptimizer = None,
        scheduler: RunnerScheduler = None,
        **kwargs,
    ) -> Dict:
        """
        Packs ``model``, ``criterion``, ``optimizer``, ``scheduler``
        and some extra info ``**kwargs`` to torch-based checkpoint.

        Args:
            model: torch model
            criterion: torch criterion
            optimizer: torch optimizer
            scheduler: torch scheduler
            **kwargs: some extra info to pack

        Returns:
            torch-based checkpoint with ``model_state_dict``,
            ``criterion_state_dict``, ``optimizer_state_dict``,
            ``scheduler_state_dict`` keys.
        """
        # @TODO: correctly handle OSS optimizer
        return {}

    def unpack_checkpoint(
        self,
        checkpoint: Dict,
        model: RunnerModel = None,
        criterion: RunnerCriterion = None,
        optimizer: RunnerOptimizer = None,
        scheduler: RunnerScheduler = None,
        **kwargs,
    ) -> None:
        """Load checkpoint from file and unpack the content to a model
        (if not None), criterion (if not None), optimizer (if not None),
        scheduler (if not None).

        Args:
            checkpoint: checkpoint to load
            model: model where should be updated state
            criterion: criterion where should be updated state
            optimizer: optimizer where should be updated state
            scheduler: scheduler where should be updated state
            kwargs: extra arguments
        """
        # @TODO: correctly handle OSS optimizer
        pass


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


class FullySharedDataParallelFairScaleEngine(SharedDataParallelFairScaleEngine):
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
