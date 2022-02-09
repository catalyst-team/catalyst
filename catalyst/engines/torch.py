# taken from https://github.com/Scitator/animus/blob/main/animus/torch/accelerate.py
from typing import Any, Callable, Dict, Optional, Union
import os

import numpy as np

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from catalyst import SETTINGS
from catalyst.core.engine import Engine
from catalyst.utils.distributed import mean_reduce

if SETTINGS.xla_required:
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.xla_multiprocessing as xmp


class CPUEngine(Engine):
    """CPU-based engine."""

    def __init__(self, *args, **kwargs) -> None:
        """Init."""
        super().__init__(*args, cpu=True, **kwargs)


class GPUEngine(Engine):
    """Single-GPU-based engine."""

    def __init__(self, *args, **kwargs) -> None:
        """Init."""
        super().__init__(*args, cpu=False, **kwargs)


class DataParallelEngine(Engine):
    """Multi-GPU-based engine."""

    def __init__(self, *args, **kwargs) -> None:
        """Init."""
        super().__init__(*args, cpu=False, **kwargs)

    def prepare_model(self, model):
        """Overrides."""
        model = torch.nn.DataParallel(model)
        model = super().prepare_model(model)
        return model


class DistributedDataParallelEngine(Engine):
    """Distributed multi-GPU-based engine.

    Args:
        *args: args for Accelerator.__init__
        address: master node (rank 0)'s address,
            should be either the IP address or the hostname
            of node 0, for single node multi-proc training, can simply be 127.0.0.1
        port: master node (rank 0)'s free port that needs to be used for communication
            during distributed training
        world_size: the number of processes to use for distributed training.
            Should be less or equal to the number of GPUs
        workers_dist_rank: the rank of the first process to run on the node.
            It should be a number between `number of initialized processes`
            and `world_size - 1`, the other processes on the node wiil have ranks
            `# of initialized processes + 1`, `# of initialized processes + 2`, ...,
            `# of initialized processes + num_node_workers - 1`
        num_node_workers: the number of processes to launch on the node.
            For GPU training, this is recommended to be set to the number of GPUs
            on the current node so that each process can be bound to a single GPU
        process_group_kwargs: parameters for `torch.distributed.init_process_group`.
            More info here:
            https://pytorch.org/docs/stable/distributed.html#torch.distributed.init_process_group  # noqa: E501, W505
        **kwargs: kwargs for Accelerator.__init__

    """

    def __init__(
        self,
        *args,
        address: str = "127.0.0.1",
        port: Union[str, int] = 2112,
        world_size: Optional[int] = None,
        workers_dist_rank: int = 0,
        num_node_workers: Optional[int] = None,
        process_group_kwargs: Dict[str, Any] = None,
        **kwargs
    ):
        """Init."""
        self._address = os.environ.get("MASTER_ADDR", address)
        self._port = os.environ.get("MASTER_PORT", port)
        self._num_local_workers = num_node_workers or torch.cuda.device_count() or 1
        self._workers_global_rank = workers_dist_rank
        self._world_size = world_size or self._num_local_workers
        self._process_group_kwargs = process_group_kwargs or {}
        self._args = args
        self._kwargs = kwargs

    def spawn(self, fn: Callable, *args, **kwargs):
        """Spawns processes with specified ``fn`` and ``args``/``kwargs``.

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
        return mp.spawn(
            fn,
            args=(self._world_size,),
            nprocs=self._num_local_workers,
            join=True,
        )

    def setup(self, local_rank: int, world_size: int):
        """Initialize DDP variables and processes if required.

        Args:
            local_rank: process rank. Default is `-1`.
            world_size: number of devices in netwok to expect for train.
                Default is `1`.
        """
        process_group_kwargs = {
            "backend": "nccl",
            "world_size": world_size,
            **self._process_group_kwargs,
        }
        global_rank = self._workers_global_rank + local_rank

        os.environ["MASTER_ADDR"] = str(self._address)
        os.environ["MASTER_PORT"] = str(self._port)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["RANK"] = str(global_rank)
        os.environ["LOCAL_RANK"] = str(local_rank)
        dist.init_process_group(**process_group_kwargs)
        super().__init__(self, *self._args, **self._kwargs)

    def cleanup(self):
        """Cleans DDP variables and processes."""
        dist.destroy_process_group()

    def mean_reduce_ddp_metrics(self, metrics: Dict) -> Dict:
        """Syncs ``metrics`` over ``world_size`` in the distributed mode."""
        metrics = {
            k: mean_reduce(
                torch.tensor(v, device=self.device),
                world_size=self.state.num_processes,
            )
            for k, v in metrics.items()
        }
        return metrics


class DistributedXLAEngine(Engine):
    """Distributed XLA-based engine."""

    def __init__(self, *args, **kwargs):
        """Init."""
        self._args = args
        self._kwargs = kwargs

    def spawn(self, fn: Callable, *args, **kwargs):
        """Spawns processes with specified ``fn`` and ``args``/``kwargs``.

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
        world_size: int = 8
        return xmp.spawn(fn, args=(world_size,), nprocs=world_size, start_method="fork")

    def setup(self, local_rank: int, world_size: int):
        """Initialize DDP variables and processes if required.

        Args:
            local_rank: process rank. Default is `-1`.
            world_size: number of devices in netwok to expect for train.
                Default is `1`.
        """
        super().__init__(self, *self._args, **self._kwargs)

    def mean_reduce_ddp_metrics(self, metrics: Dict) -> Dict:
        """Syncs ``metrics`` over ``world_size`` in the distributed mode."""
        metrics = {
            k: xm.mesh_reduce(k, v.item() if isinstance(v, torch.Tensor) else v, np.mean)
            for k, v in metrics.items()
        }
        return metrics


__all__ = [
    CPUEngine,
    GPUEngine,
    DataParallelEngine,
    DistributedDataParallelEngine,
    DistributedXLAEngine,
]
