# taken from https://github.com/Scitator/animus/blob/main/animus/torch/accelerate.py
from typing import Callable, Dict
import os

import numpy as np

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from catalyst import SETTINGS
from catalyst.core.engine import IEngine

if SETTINGS.xla_required:
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.xla_multiprocessing as xmp


class CPUEngine(IEngine):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, device_placement=False, cpu=True, **kwargs)


class GPUEngine(IEngine):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, cpu=False, **kwargs)


class DataParallelEngine(GPUEngine):
    def prepare_model(self, model):
        model = torch.nn.DataParallel(model)
        return super().prepare_model(model)


class DistributedDataParallelEngine(IEngine):
    @staticmethod
    def spawn(fn: Callable):
        world_size: int = torch.cuda.device_count()
        return mp.spawn(fn, args=(world_size,), nprocs=world_size, join=True,)

    @staticmethod
    def setup(local_rank: int, world_size: int):
        process_group_kwargs = {
            "backend": "nccl",
            "world_size": world_size,
        }
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["RANK"] = str(local_rank)
        os.environ["LOCAL_RANK"] = str(local_rank)
        dist.init_process_group(**process_group_kwargs)

    @staticmethod
    def cleanup():
        dist.destroy_process_group()

    def _sum_reduce(self, tensor: torch.Tensor) -> torch.Tensor:
        cloned = tensor.clone()
        dist.all_reduce(cloned, dist.ReduceOp.SUM)
        return cloned

    def _mean_reduce(self, tensor: torch.Tensor) -> torch.Tensor:
        reduced = self._sum_reduce(tensor) / self.state.num_processes
        return reduced

    def mean_reduce_ddp_metrics(self, metrics: Dict):
        metrics = {
            k: self._mean_reduce(torch.tensor(v, device=self.device)) for k, v in metrics.items()
        }
        return metrics


class DistributedXLAEngine(IEngine):
    @staticmethod
    def spawn(fn: Callable):
        world_size: int = 8
        xmp.spawn(fn, args=(world_size,), nprocs=world_size, start_method="fork")

    def mean_reduce_ddp_metrics(self, metrics: Dict):
        metrics = {
            k: xm.mesh_reduce(k, v.item() if isinstance(v, torch.Tensor) else v, np.mean)
            for k, v in metrics.items()
        }
        return metrics
