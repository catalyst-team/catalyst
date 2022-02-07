# taken from https://github.com/Scitator/animus/blob/main/animus/torch/accelerate.py

from typing import Callable, Dict

import numpy as np

from accelerate import Accelerator
from accelerate.state import DistributedType
import torch

from catalyst import SETTINGS
from catalyst.utils.distributed import mean_reduce

if SETTINGS.xla_required:
    import torch_xla.core.xla_model as xm


class Engine(Accelerator):
    """
    An abstraction that syncs experiment run with
    different hardware-specific configurations.
    - CPU
    - GPU
    - DataParallel (deepspeed, torch)
    - AMP (deepspeed, torch)
    - DDP (deepspeed, torch)
    - XLA

    Please check out implementations for more details:
        - :py:mod:`catalyst.engines.torch.CPUEngine`
        - :py:mod:`catalyst.engines.torch.GPUEngine`
        - :py:mod:`catalyst.engines.torch.DataParallelEngine`
        - :py:mod:`catalyst.engines.torch.DistributedDataParallelEngine`
        - :py:mod:`catalyst.engines.torch.DistributedXLAEngine`
    """

    @property
    def is_ddp(self):
        """Boolean flag for distributed type."""
        return self.distributed_type != DistributedType.NO

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
        return fn(*args, **kwargs)

    def setup(self, local_rank: int, world_size: int):
        """Initialize DDP variables and processes if required.

        Args:
            local_rank: process rank. Default is `-1`.
            world_size: number of devices in netwok to expect for train.
                Default is `1`.
        """
        pass

    def cleanup(self):
        """Cleans DDP variables and processes."""
        pass

    def mean_reduce_ddp_metrics(self, metrics: Dict) -> Dict:
        """Syncs ``metrics`` over ``world_size`` in the distributed mode."""
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


__all__ = ["Engine"]
