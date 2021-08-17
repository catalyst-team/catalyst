from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Union
import copy
import os

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel

from catalyst.core.engine import IEngine
from catalyst.typing import (
    Device,
    Model,
    Optimizer,
    RunnerCriterion,
    RunnerModel,
    RunnerOptimizer,
    RunnerScheduler,
)
from catalyst.utils.distributed import ddp_reduce
from catalyst.utils.torch import (
    any2device,
    load_checkpoint,
    pack_checkpoint,
    save_checkpoint,
    unpack_checkpoint,
)


class DeviceEngine(IEngine):
    """Single training device engine.

    Args:
        device: use device, default is `"cpu"`.

    Examples:

    .. code-block:: python

        from catalyst import dl

        runner = dl.SupervisedRunner()
        runner.train(
            engine=dl.DeviceEngine("cuda:1"),
            ...
        )

    .. code-block:: python

        from catalyst import dl

        class MyRunner(dl.IRunner):
            # ...
            def get_engine(self):
                return dl.DeviceEngine("cuda:1")
            # ...

    .. code-block:: yaml

        args:
            logs: ...

        model:
            _target_: ...
            ...

        engine:
            _target_: DeviceEngine
            device: cuda:1

        stages:
            ...

    """

    def __init__(self, device: str = None):
        """Init."""
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._device = device

    def __repr__(self) -> str:  # noqa: D105
        return f"{self.__class__.__name__}(device='{self._device}')"

    @property
    def device(self) -> Device:
        """Pytorch device."""
        return self._device

    @property
    def rank(self) -> int:
        """Process rank for distributed training."""
        return -1

    @property
    def world_size(self) -> int:
        """Process world size for distributed training."""
        return 1

    @property
    def backend(self) -> Optional[str]:
        return None

    def sync_device(
        self, tensor_or_module: Union[Dict, List, Tuple, np.ndarray, torch.Tensor, nn.Module]
    ) -> Union[Dict, List, Tuple, torch.Tensor, nn.Module]:
        """Moves ``tensor_or_module`` to Engine's deivce."""
        return any2device(tensor_or_module, device=self.device)

    def sync_tensor(self, tensor: torch.Tensor, mode: str) -> torch.Tensor:
        """Syncs ``tensor`` over ``world_size`` in distributed mode."""
        return tensor

    def sync_metrics(self, metrics: Dict) -> Dict:
        return metrics

    def init_components(
        self, model_fn=None, criterion_fn=None, optimizer_fn=None, scheduler_fn=None,
    ):
        """Inits the runs components."""
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

    def deinit_components(self, runner=None):
        """Deinits the runs components."""
        pass

    def zero_grad(self, loss: torch.Tensor, model: Model, optimizer: Optimizer) -> None:
        """Abstraction over ``model.zero_grad()`` step."""
        model.zero_grad()

    def backward_loss(self, loss: torch.Tensor, model: Model, optimizer: Optimizer) -> None:
        """Abstraction over ``loss.backward()`` step."""
        loss.backward()

    def optimizer_step(self, loss: torch.Tensor, model: Model, optimizer: Optimizer) -> None:
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
        return pack_checkpoint(
            model=model, criterion=criterion, optimizer=optimizer, scheduler=scheduler, **kwargs
        )

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
        unpack_checkpoint(
            checkpoint=checkpoint,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
        )

    def save_checkpoint(self, checkpoint: Mapping[str, Any], path: str):
        """Saves checkpoint to a file.

        Args:
            checkpoint: data to save.
            path: filepath where checkpoint should be stored.
        """
        save_checkpoint(checkpoint=checkpoint, path=path)

    def load_checkpoint(self, path: str):
        """Load checkpoint from path.

        Args:
            path: checkpoint file to load

        Returns:
            loaded checkpoint
        """
        return load_checkpoint(path=path)


class DataParallelEngine(DeviceEngine):
    """MultiGPU training device engine.

    Examples:

    .. code-block:: python

        from catalyst import dl

        runner = dl.SupervisedRunner()
        runner.train(
            engine=dl.DataParallelEngine(),
            ...
        )

    .. code-block:: python

        from catalyst import dl

        class MyRunner(dl.IRunner):
            # ...
            def get_engine(self):
                return dl.DataParallelEngine()
            # ...

    .. code-block:: yaml

        args:
            logs: ...

        model:
            _target_: ...
            ...

        engine:
            _target_: DataParallelEngine

        stages:
            ...

    """

    def __init__(self):
        """Init"""
        super().__init__(f"cuda:{torch.cuda.current_device()}")
        self.device_count = torch.cuda.device_count()

    def __repr__(self) -> str:  # noqa: D105
        return f"{self.__class__.__name__}(device_count={self.device_count})"

    def init_components(
        self, model_fn=None, criterion_fn=None, optimizer_fn=None, scheduler_fn=None,
    ):
        """Inits the runs components."""
        model = model_fn()
        model = self.sync_device(model)

        if isinstance(model, nn.Module):
            model = nn.DataParallel(model)
        elif isinstance(model, dict):
            model = {k: nn.DataParallel(v) for k, v in model.items()}
        else:
            raise ValueError("Model should be ``nn.Module`` or ``dict``")

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
    """Distributed MultiGPU training device engine.

    Args:
        address: address to use for backend.
        port: port to use for backend.
        ddp_kwargs: parameters for `torch.nn.parallel.DistributedDataParallel`.
            More info here:
            https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel
        process_group_kwargs: parameters for `torch.distributed.init_process_group`.
            More info here:
            https://pytorch.org/docs/stable/distributed.html#torch.distributed.init_process_group

    Examples:

    .. code-block:: python

        from catalyst import dl

        runner = dl.SupervisedRunner()
        runner.train(
            engine=dl.DistributedDataParallelEngine(),
            ...
        )

    .. code-block:: python

        from catalyst import dl

        class MyRunner(dl.IRunner):
            # ...
            def get_engine(self):
                return dl.DistributedDataParallelEngine(
                    address="0.0.0.0",
                    port=23234,
                    ddp_kwargs={"find_unused_parameters": False},
                    process_group_kwargs={"backend": "nccl"},
                )
            # ...

    .. code-block:: yaml

        args:
            logs: ...

        model:
            _target_: ...
            ...

        engine:
            _target_: DistributedDataParallelEngine
            address: 0.0.0.0
            port: 23234
            ddp_kwargs:
                find_unused_parameters: false
            process_group_kwargs:
                backend: nccl

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
        self._device = None

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

        self._backend = self.process_group_kwargs["backend"]
        self._world_size = self.process_group_kwargs["world_size"]

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
    def backend(self) -> Optional[str]:
        return self._backend

    def barrier(self) -> None:
        dist.barrier()

    def spawn(self, fn: Callable, *args: Any, **kwargs: Any) -> None:
        return torch.multiprocessing.spawn(
            fn, args=(self._world_size,), nprocs=self._world_size, join=True,
        )

    def setup_process(self, rank: int = -1, world_size: int = 1):
        """Initialize DDP variables and processes.

        Args:
            rank: process rank. Default is `-1`.
            world_size: number of devices in netwok to expect for train. Default is `1`.
        """
        self._rank = rank
        self._world_size = world_size
        torch.cuda.set_device(int(self._rank))
        self._device = f"cuda:{int(self._rank)}"

        self.process_group_kwargs["rank"] = rank
        self.process_group_kwargs["world_size"] = world_size

        os.environ["RANK"] = str(self._rank)
        os.environ["LOCAL_RANK"] = str(self._rank)
        os.environ["WORLD_SIZE"] = str(self._world_size)
        os.environ["MASTER_ADDR"] = str(self.address)
        os.environ["MASTER_PORT"] = str(self.port)
        dist.init_process_group(**self.process_group_kwargs)

    def cleanup_process(self):
        """Clean DDP variables and processes."""
        dist.barrier()
        dist.destroy_process_group()

    def sync_tensor(self, tensor: torch.Tensor, mode: str) -> torch.Tensor:
        """Syncs ``tensor`` over ``world_size`` in distributed mode.

        Args:
            tensor: tensor to sync across the processes.
            mode: tensor synchronization type,
                should be one of 'sum' or 'mean'.
                Default is 'mean'.

        Returns:
            torch.Tensor with synchronized values.
        """
        return ddp_reduce(tensor, mode, self.world_size)

    def sync_metrics(self, metrics: Dict) -> Dict:
        metrics = {
            k: self.sync_tensor(torch.tensor(v, device=self.device), "mean")
            for k, v in metrics.items()
        }
        return metrics

    def init_components(
        self, model_fn=None, criterion_fn=None, optimizer_fn=None, scheduler_fn=None,
    ):
        """Inits the runs components."""
        if "device_ids" not in self.ddp_kwargs:
            self.ddp_kwargs["device_ids"] = [self._device]

        # model
        model = model_fn()
        model = self.sync_device(model)
        if isinstance(model, nn.Module):
            model = DistributedDataParallel(model, **self.ddp_kwargs)
        elif isinstance(model, dict):
            model = {k: DistributedDataParallel(v, **self.ddp_kwargs) for k, v in model.items()}
        else:
            raise ValueError("Model should be ``nn.Module`` or ``dict``")
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


__all__ = ["DeviceEngine", "DataParallelEngine", "DistributedDataParallelEngine"]
