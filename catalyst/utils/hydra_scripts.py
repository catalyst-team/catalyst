from typing import Callable
import logging
import subprocess
import sys
import warnings

from omegaconf import DictConfig

import torch
import torch.distributed

from catalyst.utils.distributed import (
    get_distributed_env,
    get_distributed_params,
)

logger = logging.getLogger(__name__)


def distributed_cmd_run(
    worker_fn: Callable, distributed: bool = True, cfg: DictConfig = None
) -> None:
    """
    Distributed run

    Args:
        worker_fn: (Callable) worker fn to run in distributed mode
        distributed: (bool) distributed flag
        cfg: (DictConfig) config

    """
    distributed_params = get_distributed_params()
    local_rank = distributed_params["local_rank"]
    world_size = distributed_params["world_size"]

    if distributed and torch.distributed.is_initialized():
        warnings.warn(
            "Looks like you are trying to call distributed setup twice, "
            "switching to normal run for correct distributed training."
        )

    if (
        not distributed
        or torch.distributed.is_initialized()
        or world_size <= 1
    ):
        worker_fn(cfg)
    elif local_rank is not None:
        torch.cuda.set_device(int(local_rank))

        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        worker_fn(cfg)
    else:
        workers = []
        try:
            for local_rank in range(torch.cuda.device_count()):
                rank = distributed_params["start_rank"] + local_rank
                env = get_distributed_env(local_rank, rank, world_size)
                cmd = [sys.executable] + sys.argv.copy()
                workers.append(subprocess.Popen(cmd, env=env))
            for worker in workers:
                worker.wait()
        finally:
            for worker in workers:
                worker.kill()


__all__ = ["distributed_cmd_run"]
