# taken from https://github.com/Scitator/animus/blob/main/animus/torch/accelerate.py
from typing import Callable, Dict
import os

import numpy as np

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from catalyst import SETTINGS
from catalyst.core.engine import IEngine
from catalyst.utils.distributed import mean_reduce

if SETTINGS.xla_required:
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.xla_multiprocessing as xmp


class CPUEngine(IEngine):
    """CPU-based engine."""

    def __init__(self, *args, **kwargs) -> None:
        """Init."""
        super().__init__(*args, cpu=True, **kwargs)


class GPUEngine(IEngine):
    """Single-GPU-based engine."""

    def __init__(self, *args, **kwargs) -> None:
        """Init."""
        super().__init__(*args, cpu=False, **kwargs)


class DeviceEngine(IEngine):
    """Singe-device engine."""

    def __init__(self, *args, **kwargs) -> None:
        """Init."""
        super().__init__(*args, cpu=not torch.cuda.is_available(), **kwargs)


class DataParallelEngine(GPUEngine):
    """Multi-GPU-based engine."""

    def prepare_model(self, model):
        """Overrides."""
        model = torch.nn.DataParallel(model)
        return super().prepare_model(model)


class DistributedDataParallelEngine(IEngine):
    """Distributed multi-GPU-based engine."""

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
        world_size: int = torch.cuda.device_count()
        return mp.spawn(
            fn,
            args=(world_size,),
            nprocs=world_size,
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
        }
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["RANK"] = str(local_rank)
        os.environ["LOCAL_RANK"] = str(local_rank)
        dist.init_process_group(**process_group_kwargs)
        super().__init__(self, *self._args, **self._kwargs)

    def cleanup(self):
        """Cleans DDP variables and processes."""
        dist.destroy_process_group()

    def mean_reduce_ddp_metrics(self, metrics: Dict):
        """Syncs ``metrics`` over ``world_size`` in the distributed mode."""
        metrics = {
            k: mean_reduce(
                torch.tensor(v, device=self.device),
                world_size=self.state.num_processes,
            )
            for k, v in metrics.items()
        }
        return metrics


class DistributedXLAEngine(IEngine):
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

    def mean_reduce_ddp_metrics(self, metrics: Dict):
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
