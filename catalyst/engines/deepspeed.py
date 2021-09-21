from typing import Any, Callable, Dict, Optional, Union
import copy
import os

import torch
import torch.distributed as dist

from catalyst.engines.torch import DeviceEngine
from catalyst.settings import SETTINGS
from catalyst.utils.distributed import ddp_reduce

if SETTINGS.deepspeed_required:
    import deepspeed


class DistributedDataParallelDeepSpeedEngine(DeviceEngine):
    """Distributed DeepSpeed MultiGPU training device engine.

    Args:
        address: address to use for backend.
        port: port to use for backend.
        process_group_kwargs: parameters for `torch.distributed.init_process_group`.
            More info here:
            https://pytorch.org/docs/stable/distributed.html#torch.distributed.init_process_group
        deepspeed_kwargs: parameters for `deepspeed.initialize`.
            More info here: https://deepspeed.readthedocs.io/en/latest/initialize.html
        train_batch_size: shortcut for train batch size for deepspeed scaling (default: 256)
            for proper configuration, please use deepspeed_kwargs['config'] instead

    Examples:

    .. code-block:: python

        from catalyst import dl

        runner = dl.SupervisedRunner()
        runner.train(
            engine=dl.DistributedDataParallelDeepSpeedEngine(),
            ...
        )

    .. code-block:: python

        from catalyst import dl

        class MyRunner(dl.IRunner):
            # ...
            def get_engine(self):
                return dl.DistributedDataParallelDeepSpeedEngine(
                    address="0.0.0.0",
                    port=23234,
                    process_group_kwargs={"port": 12345},
                    deepspeed_kwargs={"config": {"train_batch_size": 64}}
                )
            # ...

    .. code-block:: yaml

        args:
            logs: ...

        model:
            _target_: ...
            ...

        engine:
            _target_: DistributedDataParallelDeepSpeedEngine
            address: 0.0.0.0
            port: 23234
            process_group_kwargs:
                port: 12345
            deepspeed_kwargs:
                config:
                    train_batch_size: 64

        stages:
            ...

    """

    def __init__(
        self,
        address: str = None,
        port: Union[str, int] = None,
        process_group_kwargs: Dict[str, Any] = None,
        deepspeed_kwargs: Dict[str, Any] = None,
        train_batch_size: int = 256,
    ):
        """Init."""
        super().__init__()
        self.address = address or "localhost"
        self.port = port or 12345
        self._rank = 0
        self._device = None

        if process_group_kwargs is None:
            process_group_kwargs = {}
        self.process_group_kwargs = copy.deepcopy(process_group_kwargs)

        # add missing arguments
        if "backend" not in self.process_group_kwargs:
            self.process_group_kwargs["dist_backend"] = "nccl"

        self._backend = self.process_group_kwargs["dist_backend"]
        self._world_size = (
            self.process_group_kwargs.get("world_size", None) or torch.cuda.device_count()
        )
        self.deepspeed_kwargs = deepspeed_kwargs or {}
        self.deepspeed_kwargs["config"] = self.deepspeed_kwargs.get("config", {})
        self.deepspeed_kwargs["config"]["train_batch_size"] = self.deepspeed_kwargs["config"].get(
            "train_batch_size", train_batch_size
        )

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
    def backend(self) -> Optional[str]:
        """String identifier for distributed backend."""
        return self._backend

    def barrier(self) -> None:
        """
        Synchronizes all processes.

        This collective blocks processes until the all runs enter the function.
        """
        dist.barrier()

    def spawn(self, fn: Callable, *args: Any, **kwargs: Any) -> None:
        """Spawns abstraction for``nprocs`` creation with specified ``fn`` and ``args``/``kwargs``.

        Args:
            fn (function): Function is called as the entrypoint of the
                spawned process. This function must be defined at the top
                level of a module so it can be pickled and spawned. This
                is a requirement imposed by multiprocessing.
                The function is called as ``fn(i, *args)``, where ``i`` is
                the process index and ``args`` is the passed through tuple
                of arguments.
            *args: Arguments passed to spawn method.
            **kwargs: Keyword-arguments passed to spawn method.

        Returns:
            wrapped function (if needed).
        """
        return torch.multiprocessing.spawn(
            fn, args=(self._world_size,), nprocs=self._world_size, join=True
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
        torch.cuda.set_device(int(self._rank))
        self._device = f"cuda:{int(self._rank)}"

        os.environ["RANK"] = str(self._rank)
        os.environ["LOCAL_RANK"] = str(self._rank)
        os.environ["WORLD_SIZE"] = str(self._world_size)
        os.environ["MASTER_ADDR"] = str(self.address)
        os.environ["MASTER_PORT"] = str(self.port)
        deepspeed.init_distributed(**self.process_group_kwargs)

    def cleanup_process(self):
        """Clean DDP variables and processes."""
        dist.barrier()
        dist.destroy_process_group()

    # @TODO: add all_gather
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

    def init_components(
        self, model_fn=None, criterion_fn=None, optimizer_fn=None, scheduler_fn=None
    ):
        """Inits the runs components."""
        model = model_fn()
        model = self.sync_device(model)

        criterion = criterion_fn()
        criterion = self.sync_device(criterion)

        optimizer = optimizer_fn()
        optimizer = self.sync_device(optimizer)

        scheduler = scheduler_fn()
        scheduler = self.sync_device(scheduler)

        model, optimizer, _, scheduler = deepspeed.initialize(
            model=model, optimizer=optimizer, lr_scheduler=scheduler, **self.deepspeed_kwargs
        )

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
