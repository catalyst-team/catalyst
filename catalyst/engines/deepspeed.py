from typing import Any, Dict, Optional, Union
import os

import torch

from catalyst.engines.torch import DistributedDataParallelEngine
from catalyst.settings import SETTINGS

if SETTINGS.deepspeed_required:
    import deepspeed


class DistributedDataParallelDeepSpeedEngine(DistributedDataParallelEngine):
    """Distributed DeepSpeed MultiGPU training device engine.

    Args:
        address: master node (rank 0)'s address, should be either the IP address or the hostname
            of node 0, for single node multi-proc training, can simply be 127.0.0.1
        port: master node (rank 0)'s free port that needs to be used for communication
            during distributed training
        world_size: the number of processes to use for distributed training.
            Should be less or equal to the number of GPUs
        workers_dist_rank: the rank of the first process to run on the node.
            It should be a number between `number of initialized processes` and `world_size - 1`,
            the other processes on the node wiil have ranks `# of initialized processes + 1`,
            `# of initialized processes + 2`, ...,
            `# of initialized processes + num_node_workers - 1`
        num_node_workers: the number of processes to launch on the node.
            For GPU training, this is recommended to be set to the number of GPUs
            on the current node so that each process can be bound to a single GPU
        process_group_kwargs: parameters for `torch.distributed.init_process_group`.
            More info here:
            https://pytorch.org/docs/stable/distributed.html#torch.distributed.init_process_group
        train_batch_size: shortcut for train batch size for deepspeed scaling (default: 256)
            for proper configuration, please use deepspeed_kwargs['config'] instead
        deepspeed_kwargs: parameters for `deepspeed.initialize`.
            More info here: https://deepspeed.readthedocs.io/en/latest/initialize.html

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
        address: str = "127.0.0.1",
        port: Union[str, int] = 2112,
        world_size: Optional[int] = None,
        workers_dist_rank: int = 0,
        num_node_workers: Optional[int] = None,
        process_group_kwargs: Dict[str, Any] = None,
        train_batch_size: int = 256,
        deepspeed_kwargs: Dict[str, Any] = None,
    ):
        """Init."""
        super().__init__(
            address=address,
            port=port,
            world_size=world_size,
            workers_dist_rank=workers_dist_rank,
            num_node_workers=num_node_workers,
            process_group_kwargs=process_group_kwargs,
        )

        process_group_kwargs = process_group_kwargs or {}
        self.process_group_kwargs = {
            "dist_backend": "nccl",
            **process_group_kwargs,
        }
        self._backend = self.process_group_kwargs["dist_backend"]

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

    def setup_process(self, rank: int = -1, world_size: int = 1):
        """Initialize DDP variables and processes.

        Args:
            rank: process rank. Default is `-1`.
            world_size: number of devices in netwok to expect for train.
                Default is `1`.
        """
        self._rank = self.workers_global_rank + rank

        if torch.cuda.is_available():
            torch.cuda.set_device(int(rank))
            self._device = f"cuda:{int(rank)}"

        os.environ["MASTER_ADDR"] = str(self.address)
        os.environ["MASTER_PORT"] = str(self.port)
        os.environ["WORLD_SIZE"] = str(self._world_size)
        os.environ["RANK"] = str(self._rank)
        os.environ["LOCAL_RANK"] = str(rank)
        deepspeed.init_distributed(**self.process_group_kwargs)

    def init_components(
        self, model_fn=None, criterion_fn=None, optimizer_fn=None, scheduler_fn=None
    ):
        """Inits the runs components."""
        model = model_fn()
        model = self.sync_device(model)

        criterion = criterion_fn()
        criterion = self.sync_device(criterion)

        optimizer = optimizer_fn(model)
        optimizer = self.sync_device(optimizer)

        scheduler = scheduler_fn(optimizer)
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
