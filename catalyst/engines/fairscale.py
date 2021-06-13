from typing import Any, Dict, Union
import copy
import math
import os
import warnings

import torch
import torch.cuda.amp as amp
import torch.distributed as dist

from catalyst.engines.torch import DeviceEngine
from catalyst.settings import SETTINGS
from catalyst.typing import RunnerCriterion, RunnerModel, RunnerOptimizer, RunnerScheduler
from catalyst.utils.distributed import mean_reduce, sum_reduce

if SETTINGS.fairscale_required:
    from fairscale.nn import Pipe
    from fairscale.nn.data_parallel import (
        FullyShardedDataParallel as FSDP,
        ShardedDataParallel as ShardedDDP,
    )
    from fairscale.optim import OSS
    from fairscale.optim.grad_scaler import ShardedGradScaler


def _generate_balance(num_devices: int, num_layers: int):
    balance = []
    layers_assigned = 0
    for i in range(num_devices):
        x = (num_layers - layers_assigned) / (num_devices - i)
        if x.is_integer():
            balance.append(int(x))
            layers_assigned += x
        else:
            balance.append(math.ceil(x))
            layers_assigned += math.ceil(x)
    return balance


class PipelineParallelFairScaleEngine(DeviceEngine):
    """FairScale multi-gpu training device engine.

    Args:
        pipe_kwargs: parameters for `fairscale.nn.Pipe`.
            Docs for `fairscale.nn.Pipe`:
            https://fairscale.readthedocs.io/en/latest/api/nn/pipe.html

    Examples:

    .. code-block:: python

        from catalyst import dl

        runner = dl.SupervisedRunner()
        runner.train(
            engine=dl.PipelineParallelFairScaleEngine(),
            ...
        )

    .. code-block:: python

        from catalyst import dl

        class MyRunner(dl.IRunner):
            # ...
            def get_engine(self):
                return dl.PipelineParallelFairScaleEngine(
                    pipe_kwargs={"balance": [3, 1]},
                )
            # ...

    .. code-block:: yaml

        args:
            logs: ...

        model:
            _target_: ...
            ...

        engine:
            _target_: PipelineParallelFairScaleEngine
            pipe_kwargs:
                balance: [3, 1]

        stages:
            ...

    """

    def __init__(self, pipe_kwargs: Dict[str, Any] = None):
        """Init."""
        super().__init__(f"cuda:{torch.cuda.current_device()}")
        self.device_count = torch.cuda.device_count()
        assert self.device_count > 0
        self.pipe_kwargs = pipe_kwargs or {}

    def __repr__(self) -> str:  # noqa: D105
        return f"{self.__class__.__name__}(device_count={self.device_count})"

    def init_components(
        self, model_fn=None, criterion_fn=None, optimizer_fn=None, scheduler_fn=None,
    ):
        """Inits the runs components."""

        model = model_fn()
        # model = self.sync_device(model)

        if "balance" not in self.pipe_kwargs:
            warnings.warn(
                "With FairScale Pipe setup, "
                "you need to specify ``balance`` under ``pipe_kwargs``. "
                "Generating balance automatically. (Experimental feature)"
            )
            self.pipe_kwargs["balance"] = _generate_balance(self.device_count, len(model))
        pipe_model = Pipe(model, **self.pipe_kwargs)
        del model

        # criterion
        criterion = criterion_fn()
        # criterion = self.sync_device(criterion)

        # optimizer
        optimizer = optimizer_fn(pipe_model)
        # optimizer = self.sync_device(optimizer)

        # scheduler
        scheduler = scheduler_fn()
        # scheduler = self.sync_device(scheduler)
        return pipe_model, criterion, optimizer, scheduler

    def deinit_components(self, runner):
        """Deinits the runs components.
        In distributed mode should destroy process group.
        """
        del runner.model

    def zero_grad(self, loss, model, optimizer) -> None:
        """Abstraction over ``model.zero_grad()`` step."""
        optimizer.zero_grad()

    def backward_loss(self, loss, model, optimizer) -> None:
        """Abstraction over ``loss.backward()`` step."""
        loss.backward()

    def optimizer_step(self, loss, model, optimizer) -> None:
        """Abstraction over ``optimizer.step()`` step."""
        optimizer.step()


class SharedDataParallelFairScaleEngine(DeviceEngine):
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

    Examples:

    .. code-block:: python

        from catalyst import dl

        runner = dl.SupervisedRunner()
        runner.train(
            engine=dl.SharedDataParallelFairScaleEngine(),
            ...
        )

    .. code-block:: python

        from catalyst import dl

        class MyRunner(dl.IRunner):
            # ...
            def get_engine(self):
                return dl.SharedDataParallelFairScaleEngine(
                    address="0.0.0.0",
                    port=23234,
                    ddp_kwargs={"find_unused_parameters": False},
                    process_group_kwargs={"port": 12345},
                )
            # ...

    .. code-block:: yaml

        args:
            logs: ...

        model:
            _target_: ...
            ...

        engine:
            _target_: SharedDataParallelFairScaleEngine
            address: 0.0.0.0
            port: 23234
            ddp_kwargs:
                find_unused_parameters: false
            process_group_kwargs:
                port: 12345

        stages:
            ...

    """

    def __init__(
        self,
        address: str = None,
        port: Union[str, int] = None,
        ddp_kwargs: Dict[str, Any] = None,
        process_group_kwargs: Dict[str, Any] = None,
    ):
        """Init."""
        super().__init__()
        self.address = address or "localhost"
        self.port = port or 12345
        self._rank = 0
        self.device = None

        if ddp_kwargs is None:
            ddp_kwargs = {}
        self.ddp_kwargs = copy.deepcopy(ddp_kwargs)

        if process_group_kwargs is None:
            process_group_kwargs = {}
        self.process_group_kwargs = copy.deepcopy(process_group_kwargs)
        # add missing arguments
        if "backend" not in self.process_group_kwargs:
            self.process_group_kwargs["backend"] = "nccl"
        if "world_size" not in self.process_group_kwargs:
            self.process_group_kwargs["world_size"] = torch.cuda.device_count()

        self._world_size = (
            self.process_group_kwargs.get("world_size", None) or torch.cuda.device_count()
        )

    def __repr__(self):  # noqa: D105
        return (
            f"{self.__class__.__name__}(address={self.address}, "
            f"port={self.port}, "
            f"ddp_kwargs={self.ddp_kwargs}, "
            f"process_group_kwargs={self.process_group_kwargs})"
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

        self.process_group_kwargs["rank"] = rank
        self.process_group_kwargs["world_size"] = world_size
        os.environ["MASTER_ADDR"] = str(self.address)
        os.environ["MASTER_PORT"] = str(self.port)

        dist.init_process_group(**self.process_group_kwargs)

        torch.cuda.set_device(int(self._rank))
        self.device = f"cuda:{int(self._rank)}"

    def cleanup_process(self):
        """Clean DDP variables and processes."""
        dist.barrier()
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

        optimizer = optimizer_fn(model)
        optimizer = self.sync_device(optimizer)

        optimizer = OSS(model.parameters(), optim=optimizer.__class__, **optimizer.defaults)
        model = ShardedDDP(model, optimizer, **self.ddp_kwargs)

        scheduler = scheduler_fn(optimizer)
        scheduler = self.sync_device(scheduler)
        return model, criterion, optimizer, scheduler

    def deinit_components(self, runner=None):
        """Deinits the runs components."""
        pass
        # self.cleanup_process()

    def zero_grad(self, loss, model, optimizer) -> None:
        """Abstraction over ``model.zero_grad()`` step."""
        model.zero_grad()

    def backward_loss(self, loss, model, optimizer) -> None:
        """Abstraction over ``loss.backward()`` step."""
        loss.backward()

    def optimizer_step(self, loss, model, optimizer) -> None:
        """Abstraction over ``optimizer.step()`` step."""
        optimizer.step()

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
            Docs for `fairscale.nn.ShardedDataParallel`:
            https://fairscale.readthedocs.io/en/latest/api/nn/sharded_ddp.html
        process_group_kwargs: parameters for `torch.distributed.init_process_group`.
            More info here:
            https://pytorch.org/docs/stable/distributed.html#torch.distributed.init_process_group
        scaler_kwargs: parameters for `fairscale.optim.grad_scaler.ShardedGradScaler`.
            Possible parameters:
            https://fairscale.readthedocs.io/en/latest/api/index.html

    Examples:

    .. code-block:: python

        from catalyst import dl

        runner = dl.SupervisedRunner()
        runner.train(
            engine=dl.SharedDataParallelFairScaleAMPEngine(),
            ...
        )

    .. code-block:: python

        from catalyst import dl

        class MyRunner(dl.IRunner):
            # ...
            def get_engine(self):
                return dl.SharedDataParallelFairScaleAMPEngine(
                    address="0.0.0.0",
                    port=23234,
                    ddp_kwargs={"find_unused_parameters": False},
                    process_group_kwargs={"port": 12345},
                    scaler_kwargs={"growth_factor": 1.5}
                )
            # ...

    .. code-block:: yaml

        args:
            logs: ...

        model:
            _target_: ...
            ...

        engine:
            _target_: SharedDataParallelFairScaleAMPEngine
            address: 0.0.0.0
            port: 23234
            ddp_kwargs:
                find_unused_parameters: false
            process_group_kwargs:
                port: 12345
            scaler_kwargs:
                growth_factor: 1.5

        stages:
            ...

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
        # @TODO: should we support scaler for each optimizer?
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
            Docs for `fairscale.nn.FullyShardedDataParallel`:
            https://fairscale.readthedocs.io/en/latest/api/nn/fsdp.html
        process_group_kwargs: parameters for `torch.distributed.init_process_group`.
            More info here:
            https://pytorch.org/docs/stable/distributed.html#torch.distributed.init_process_group

    Examples:

    .. code-block:: python

        from catalyst import dl

        runner = dl.SupervisedRunner()
        runner.train(
            engine=dl.FullySharedDataParallelFairScaleEngine(),
            ...
        )

    .. code-block:: python

        from catalyst import dl

        class MyRunner(dl.IRunner):
            # ...
            def get_engine(self):
                return dl.FullySharedDataParallelFairScaleEngine(
                    address="0.0.0.0",
                    port=23234,
                    ddp_kwargs={"find_unused_parameters": False},
                    process_group_kwargs={"port": 12345},
                )
            # ...

    .. code-block:: yaml

        args:
            logs: ...

        model:
            _target_: ...
            ...

        engine:
            _target_: FullySharedDataParallelFairScaleEngine
            address: 0.0.0.0
            port: 23234
            ddp_kwargs:
                find_unused_parameters: false
            process_group_kwargs:
                port: 12345

        stages:
            ...

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

        model = FSDP(model, **self.ddp_kwargs)

        criterion = criterion_fn()
        criterion = self.sync_device(criterion)

        optimizer = optimizer_fn(model)
        optimizer = self.sync_device(optimizer)

        scheduler = scheduler_fn(optimizer)
        scheduler = self.sync_device(scheduler)

        return model, criterion, optimizer, scheduler


__all__ = [
    "PipelineParallelFairScaleEngine",
    "SharedDataParallelFairScaleEngine",
    "SharedDataParallelFairScaleAMPEngine",
    "FullySharedDataParallelFairScaleEngine",
]
