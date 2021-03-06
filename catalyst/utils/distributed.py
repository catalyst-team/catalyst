from collections import OrderedDict
import os
import random
import socket
import subprocess

import torch
from torch import nn
import torch.distributed

from catalyst.settings import SETTINGS
from catalyst.utils.torch import get_available_gpus


def check_torch_distributed_initialized() -> bool:
    """Checks if torch.distributed is available and initialized."""
    return (
        torch.distributed.is_available() and torch.distributed.is_initialized()
    )


def check_slurm_available():
    """Checks if slurm is available."""
    return "SLURM_JOB_NUM_NODES" in os.environ and "SLURM_NODEID" in os.environ


def check_ddp_wrapped(model: nn.Module) -> bool:
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
    if check_ddp_wrapped(model):
        model = model.module
    return model


def get_rank() -> int:
    """
    Returns the rank of the current worker.

    Returns:
        int: ``rank`` if torch.distributed is initialized, otherwise ``-1``
    """
    if check_torch_distributed_initialized():
        return torch.distributed.get_rank()
    else:
        return -1


# TODO: rename
def get_slurm_params():
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
    if check_slurm_available():
        cur_node, num_nodes, master_addr, master_port = get_slurm_params()

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


def get_distributed_env(
    local_rank: int,
    rank: int,
    world_size: int,
    use_cuda_visible_devices: bool = True,
):
    """Returns environment copy with extra distributed settings.

    Args:
        local_rank: worker local rank
        rank: worker global rank
        world_size: worker world size
        use_cuda_visible_devices: boolean flag to use available GPU devices

    Returns:
        updated environment copy
    """
    env = os.environ.copy()
    env["RANK"] = str(rank)
    env["WORLD_SIZE"] = str(world_size)
    env["LOCAL_RANK"] = str(local_rank)
    if use_cuda_visible_devices:
        available_gpus = get_available_gpus()
        env["LOCAL_RANK"] = "0"
        env["CUDA_VISIBLE_DEVICES"] = str(available_gpus[local_rank])
    return env


__all__ = [
    "check_ddp_wrapped",
    "check_torch_distributed_initialized",
    "check_slurm_available",
    "get_nn_from_ddp_module",
    "get_rank",
    "get_distributed_env",
    "get_distributed_params",
    "get_slurm_params",
]
