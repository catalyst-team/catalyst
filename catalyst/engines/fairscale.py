from typing import Any, Dict, Union
import math
import warnings

import torch
import torch.cuda.amp as amp
import torch.nn as nn

from catalyst.engines.torch import DeviceEngine, DistributedDataParallelEngine
from catalyst.settings import SETTINGS
from catalyst.typing import RunnerCriterion, RunnerModel, RunnerOptimizer, RunnerScheduler

if SETTINGS.fairscale_required:
    from fairscale.nn import Pipe
    from fairscale.nn.data_parallel import FullyShardedDataParallel, ShardedDataParallel
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
                    pipe_kwargs={"balance": [3, 1]}
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
        self, model_fn=None, criterion_fn=None, optimizer_fn=None, scheduler_fn=None
    ):
        """Inits the runs components."""
        model = model_fn()

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
        # optimizer
        optimizer = optimizer_fn(pipe_model)
        # scheduler
        scheduler = scheduler_fn()

        return pipe_model, criterion, optimizer, scheduler

    # due to FairScale setup, we need to manually delete the model in the end
    def deinit_components(self, runner):
        """Deinits the runs components. In distributed mode should destroy process group."""
        # For some reasons FairScale requires to delete the Pipe model
        del runner.callbacks
        runner.callbacks = {}
        del runner.loaders
        runner.loaders = {}
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


class SharedDataParallelFairScaleEngine(DistributedDataParallelEngine):
    """Distributed FairScale MultiGPU training device engine.

    Args:
        address: address to use for backend.
        port: port to use for backend.
        sync_bn: boolean flag for batchnorm synchonization during disributed training.
            if True, applies PyTorch `convert_sync_batchnorm`_ to the model for native torch
            distributed only. Default, False.
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
    .. _convert_sync_batchnorm:
        https://pytorch.org/docs/stable/generated/torch.nn.SyncBatchNorm.html#
        torch.nn.SyncBatchNorm.convert_sync_batchnorm
    """

    def init_components(
        self, model_fn=None, criterion_fn=None, optimizer_fn=None, scheduler_fn=None
    ):
        """Inits the runs components."""
        model = model_fn()
        model = self.sync_device(model)
        if self._sync_bn:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

        criterion = criterion_fn()
        criterion = self.sync_device(criterion)

        optimizer = optimizer_fn(model)
        optimizer = self.sync_device(optimizer)

        optimizer = OSS(model.parameters(), optim=optimizer.__class__, **optimizer.defaults)
        model = ShardedDataParallel(model, optimizer, **self.ddp_kwargs)

        scheduler = scheduler_fn(optimizer)
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
        # for some reasons FairScale could not consolidate the optimizer step at 0.3.4 version
        # optimizer.consolidate_state_dict(recipient_rank=0)
        return super().pack_checkpoint(
            model=model, criterion=criterion, optimizer=None, scheduler=scheduler, **kwargs
        )


class SharedDataParallelFairScaleAMPEngine(SharedDataParallelFairScaleEngine):
    """Distributed FairScale MultiGPU training device engine.

    Args:
        address: address to use for backend.
        port: port to use for backend.
        sync_bn: boolean flag for batchnorm synchonization during disributed training.
            if True, applies PyTorch `convert_sync_batchnorm`_ to the model for native torch
            distributed only. Default, False.
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

    .. _convert_sync_batchnorm:
        https://pytorch.org/docs/stable/generated/torch.nn.SyncBatchNorm.html#
        torch.nn.SyncBatchNorm.convert_sync_batchnorm
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
        sync_bn: boolean flag for batchnorm synchonization during disributed training.
            if True, applies PyTorch `convert_sync_batchnorm`_ to the model for native torch
            distributed only. Default, False.
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

    .. _convert_sync_batchnorm:
        https://pytorch.org/docs/stable/generated/torch.nn.SyncBatchNorm.html#
        torch.nn.SyncBatchNorm.convert_sync_batchnorm
    """

    def init_components(
        self, model_fn=None, criterion_fn=None, optimizer_fn=None, scheduler_fn=None
    ):
        """Inits the runs components."""
        model = model_fn()
        model = self.sync_device(model)
        if self._sync_bn:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

        model = FullyShardedDataParallel(model, **self.ddp_kwargs)

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
