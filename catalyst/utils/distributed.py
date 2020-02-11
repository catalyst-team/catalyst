from collections import OrderedDict
import copy
import os
import socket
import subprocess
import sys
from typing import Dict, Tuple

import torch
from torch import nn
import torch.distributed

from catalyst import utils
from catalyst.utils.misc import get_default_params
from catalyst.utils.tools.typing import (
    Criterion, Device, Model, Optimizer, Scheduler
)


def get_rank() -> int:
    """
    Returns the rank of the current worker.

    Returns:
         int: ``rank`` if torch.distributed is initialized,
              otherwise ``-1``
    """
    if torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    else:
        return -1


def is_apex_available() -> bool:
    try:
        import apex  # noqa: F401
        from apex import amp  # noqa: F401
        return True
    except ImportError:
        return False


def assert_fp16_available() -> None:
    """
    Asserts for installed and available Apex FP16
    """
    assert torch.backends.cudnn.enabled, \
        "fp16 mode requires cudnn backend to be enabled."

    assert is_apex_available(), "NVidia Apex package must be installed. "  \
                                "See https://github.com/NVIDIA/apex."


def distributed_mean(value: float):
    if torch.distributed.is_initialized():
        value = torch.tensor(
            value,
            dtype=torch.float,
            device=f"cuda:{torch.cuda.current_device()}",
            requires_grad=False
        )
        torch.distributed.all_reduce(value)
        value = float(value.item() / torch.distributed.get_world_size())
    return value


def get_slurm_params():
    cmd = "scontrol show hostnames '%s'" % os.environ["SLURM_JOB_NODELIST"]
    nodes = subprocess.getoutput(cmd).split()
    num_nodes = int(os.environ["SLURM_JOB_NUM_NODES"])
    current_node = os.environ["SLURMD_NODENAME"]
    master_node = socket.gethostbyname(nodes[0])
    cur_node_idx = nodes.index(current_node)
    return cur_node_idx, num_nodes, master_node


def is_slurm_available():
    return "SLURM_JOB_NUM_NODES" in os.environ and "SLURM_NODEID" in os.environ


def get_distributed_params():
    master_addr = "127.0.0.1"
    cur_node, num_nodes = 0, 1
    if is_slurm_available():
        cur_node, num_nodes, master_addr = get_slurm_params()

    os.environ["MASTER_ADDR"] = os.getenv("MASTER_ADDR", master_addr)
    os.environ["MASTER_PORT"] = os.getenv("MASTER_PORT", "424242")

    workers_per_node = torch.cuda.device_count()
    start_rank = cur_node * workers_per_node
    world_size = num_nodes * workers_per_node

    local_rank = os.getenv("LOCAL_RANK", None)
    rank = os.getenv("RANK", None)
    local_rank, rank = [v and int(v) for v in [local_rank, rank]]
    world_size = int(os.getenv("WORLD_SIZE", world_size))

    return OrderedDict(
        local_rank=local_rank,
        start_rank=start_rank,
        rank=rank,
        world_size=world_size,
        master_addr=os.environ["MASTER_ADDR"],
        master_port=os.environ["MASTER_PORT"],
    )


def get_distributed_env(
    local_rank, rank, world_size, use_cuda_visible_devices=True
):
    env = os.environ.copy()
    env["RANK"] = str(rank)
    env["WORLD_SIZE"] = str(world_size)
    env["LOCAL_RANK"] = str(local_rank)
    available_gpus = utils.get_available_gpus()
    if use_cuda_visible_devices:
        env["LOCAL_RANK"] = "0"
        env["CUDA_VISIBLE_DEVICES"] = str(available_gpus[local_rank])
    return env


def distributed_run(data_parallel, worker_fn, *args, **kwargs):
    distributed_params = get_distributed_params()
    local_rank = distributed_params["local_rank"]
    world_size = distributed_params["world_size"]

    if data_parallel or world_size <= 1:
        worker_fn(*args, **kwargs)
    elif local_rank is not None:
        torch.cuda.set_device(int(local_rank))

        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        worker_fn(*args, **kwargs)
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


def process_components(
    model: Model,
    criterion: Criterion = None,
    optimizer: Optimizer = None,
    scheduler: Scheduler = None,
    distributed_params: Dict = None,
    device: Device = None,
) -> Tuple[Model, Criterion, Optimizer, Scheduler, Device]:
    """
    Returns the processed model, criterion, optimizer, scheduler and device

    Args:
        model (Model): torch model
        criterion (Criterion): criterion function
        optimizer (Optimizer): optimizer
        scheduler (Scheduler): scheduler
        distributed_params (dict, optional): dict with the parameters
            for distributed and FP16 methond
        device (Device, optional): device
    """
    distributed_params = distributed_params or {}
    distributed_params = copy.deepcopy(distributed_params)
    distributed_params.update(get_distributed_params())
    if device is None:
        device = utils.get_device()

    model: Model = utils.maybe_recursive_call(model, "to", device=device)

    if utils.is_wrapped_with_ddp(model):
        pass
    elif get_rank() >= 0:
        assert isinstance(model, nn.Module)
        local_rank = distributed_params.pop("local_rank", 0)
        device = f"cuda:{local_rank}"
        model = utils.maybe_recursive_call(model, "to", device=device)

        syncbn = distributed_params.pop("syncbn", False)
        use_apex = distributed_params.pop("apex", True) and is_apex_available()

        if use_apex:
            import apex
            amp_params = get_default_params(
                apex.amp.initialize, ["models", "optimizers"]
            )
            amp_params["opt_level"] = "O0"
            for dp in distributed_params:
                if dp in amp_params:
                    amp_params[dp] = distributed_params[dp]

            amp_result = apex.amp.initialize(model, optimizer, **amp_params)
            if optimizer is not None:
                model, optimizer = amp_result
            else:
                model = amp_result

            model = apex.parallel.DistributedDataParallel(model)

            if syncbn:
                model = apex.parallel.convert_syncbn_model(model)
        else:
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[local_rank], output_device=local_rank
            )
    elif torch.cuda.device_count() > 1:
        if isinstance(model, nn.Module):
            model = torch.nn.DataParallel(model)
        elif isinstance(model, dict):
            model = {k: torch.nn.DataParallel(v) for k, v in model.items()}

    model: Model = utils.maybe_recursive_call(model, "to", device=device)

    return model, criterion, optimizer, scheduler, device


__all__ = [
    "get_rank", "process_components", "distributed_mean", "is_apex_available",
    "assert_fp16_available", "distributed_run"
]
