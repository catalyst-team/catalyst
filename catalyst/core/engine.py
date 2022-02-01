# taken from https://github.com/Scitator/animus/blob/main/animus/torch/accelerate.py

from typing import Callable, Dict

from accelerate import Accelerator
from accelerate.state import DistributedType
import numpy as np

import torch

from catalyst import SETTINGS
from catalyst.utils.distributed import mean_reduce

if SETTINGS.xla_required:
    import torch_xla.core.xla_model as xm


class IEngine(Accelerator):
    @staticmethod
    def spawn(fn: Callable):
        fn()

    @staticmethod
    def setup(local_rank: int, world_size: int):
        pass

    @staticmethod
    def cleanup():
        pass

    def mean_reduce_ddp_metrics(self, metrics: Dict):
        if self.state.distributed_type in [
            DistributedType.MULTI_CPU,
            DistributedType.MULTI_GPU,
        ]:
            metrics = {
                k: mean_reduce(
                    torch.tensor(v, device=self.device),
                    world_size=self.state.num_processes,
                )
                for k, v in metrics.items()
            }
        elif self.state.distributed_type == DistributedType.TPU:
            metrics = {
                k: xm.mesh_reduce(
                    k, v.item() if isinstance(v, torch.Tensor) else v, np.mean
                )
                for k, v in metrics.items()
            }
        return metrics


__all__ = ["IEngine"]
