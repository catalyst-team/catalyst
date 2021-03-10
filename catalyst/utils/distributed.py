from collections import OrderedDict
import os
import random
import socket
import subprocess

import torch
from torch import nn
import torch.distributed

from catalyst.settings import SETTINGS


def _is_torch_distributed_initialized() -> bool:
    """Checks if torch.distributed is available and initialized."""
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def _is_slurm_available():
    """Checks if slurm is available."""
    return "SLURM_JOB_NUM_NODES" in os.environ and "SLURM_NODEID" in os.environ


def _is_ddp_wrapped(model: nn.Module) -> bool:
    """Checks whether model is wrapped with DataParallel/DistributedDataParallel."""
    parallel_wrappers = nn.DataParallel, nn.parallel.DistributedDataParallel

    # Check whether Apex is installed and if it is,
    # add Apex's DistributedDataParallel to list of checked types
    if SETTINGS.apex_required:
        from apex.parallel import DistributedDataParallel as apex_DDP

        parallel_wrappers = parallel_wrappers + (apex_DDP,)

    return isinstance(model, parallel_wrappers)


def get_nn_from_ddp_module(model: nn.Module) -> nn.Module:
    """
    Return a real model from a torch.nn.DataParallel,
    torch.nn.parallel.DistributedDataParallel, or
    apex.parallel.DistributedDataParallel.

    Args:
        model: A model, or DataParallel wrapper.

    Returns:
        A model
    """
    if _is_ddp_wrapped(model):
        model = model.module
    return model


def get_rank() -> int:
    """
    Returns the rank of the current worker.

    Returns:
        int: ``rank`` if torch.distributed is initialized, otherwise ``-1``
    """
    if _is_torch_distributed_initialized():
        return torch.distributed.get_rank()
    else:
        return -1


# TODO: rename
def _get_slurm_params():
    """Return slurm params for experiment run.

    Returns:
        tuple with current node index, number of nodes, master node
            and master port
    """
    cmd = "scontrol show hostnames '%s'" % os.environ["SLURM_JOB_NODELIST"]
    nodes = subprocess.getoutput(cmd).split()
    num_nodes = int(os.environ["SLURM_JOB_NUM_NODES"])
    current_node = os.environ["SLURMD_NODENAME"]
    master_node = socket.gethostbyname(nodes[0])
    cur_node_idx = nodes.index(current_node)
    job_id = os.environ["SLURM_JOB_ID"]
    master_port = str(5 * 10 ** 4 + int(job_id) % 10 ** 4)
    return cur_node_idx, num_nodes, master_node, master_port


# TODO: rename
def get_distributed_params():
    """Returns distributed params for experiment run.

    Returns:
        dictionary with distributed params
    """
    master_port = str(random.randint(5 * 10 ** 4, 6 * 10 ** 4))
    master_addr = "127.0.0.1"
    cur_node, num_nodes = 0, 1
    if _is_slurm_available():
        cur_node, num_nodes, master_addr, master_port = _get_slurm_params()

    os.environ["MASTER_ADDR"] = os.getenv("MASTER_ADDR", master_addr)
    os.environ["MASTER_PORT"] = os.getenv("MASTER_PORT", master_port)

    workers_per_node = torch.cuda.device_count()
    start_rank = cur_node * workers_per_node
    world_size = num_nodes * workers_per_node

    local_rank = os.getenv("LOCAL_RANK", None)
    rank = os.getenv("RANK", None)
    local_rank, rank = [v and int(v) for v in [local_rank, rank]]
    world_size = int(os.getenv("WORLD_SIZE", world_size))

    output = OrderedDict(
        local_rank=local_rank,
        start_rank=start_rank,
        rank=rank,
        world_size=world_size,
        master_addr=os.environ["MASTER_ADDR"],
        master_port=os.environ["MASTER_PORT"],
    )

    return output


__all__ = [
    "get_rank",
    "get_distributed_params",
    "get_nn_from_ddp_module",
]
