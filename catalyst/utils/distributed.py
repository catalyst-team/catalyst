from typing import Union
from collections import OrderedDict
import os
import random
import socket
import subprocess

from packaging.version import parse, Version
import torch
from torch import nn
import torch.distributed

from catalyst.utils.misc import get_fn_default_params
from catalyst.utils.torch import get_available_gpus


def check_ddp_wrapped(model: nn.Module) -> bool:
    """
    Checks whether model is wrapped with DataParallel/DistributedDataParallel.
    """
    parallel_wrappers = nn.DataParallel, nn.parallel.DistributedDataParallel

    # Check whether Apex is installed and if it is,
    # add Apex's DistributedDataParallel to list of checked types
    try:
        from apex.parallel import DistributedDataParallel as apex_DDP

        parallel_wrappers = parallel_wrappers + (apex_DDP,)
    except ImportError:
        pass

    return isinstance(model, parallel_wrappers)


def check_apex_available() -> bool:
    """Checks if apex is available."""
    env_apex = os.getenv("USE_APEX", "1") == "1"
    try:
        import apex  # noqa: F401
        from apex import amp  # noqa: F401

        return True and env_apex
    except ImportError:
        return False and env_apex


def check_amp_available() -> bool:
    """Checks if torch.amp is available."""
    return parse(torch.__version__) >= Version("1.6.0")


def check_torch_distributed_initialized() -> bool:
    """Checks if torch.distributed is available and initialized."""
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def check_slurm_available():
    """Checks if slurm is available."""
    return "SLURM_JOB_NUM_NODES" in os.environ and "SLURM_NODEID" in os.environ


def assert_fp16_available() -> None:
    """Asserts for installed and available Apex FP16."""
    assert torch.backends.cudnn.enabled, "fp16 mode requires cudnn backend to be enabled."

    assert check_apex_available(), (
        "NVidia Apex package must be installed. " "See https://github.com/NVIDIA/apex."
    )


def initialize_apex(model, optimizer=None, **engine_params):
    """
    Prepares model and optimizer for work with Nvidia Apex.

    Args:
        model: torch model
        optimizer: torch optimizer
        **engine_params: extra params for ``apex.amp.initialize``

    Returns:
        model and optimizer, wrapped with Nvidia Apex initialization
    """
    import apex

    amp_params = get_fn_default_params(apex.amp.initialize, ["models", "optimizers"])
    amp_params["opt_level"] = "O0"
    for dp in engine_params:
        if dp in amp_params:
            amp_params[dp] = engine_params[dp]

    # NVIDIA apex support only:
    #  model: nn.Module or list of modules
    #  optimizer: None, torch.Optimizer or list of optimizers
    # while key-value is preferred in the `catalyst`.
    # So if model/optimizer is a dict, convert it to lists of keys
    # and values first, and then cast it back after apex initialization
    model_keys, optimizer_keys = None, None
    if isinstance(model, dict):
        model_keys, model = list(model.keys()), list(model.values())
    if isinstance(optimizer, dict):
        optimizer_keys = list(optimizer.keys())
        optimizer = list(optimizer.values())

    amp_result = apex.amp.initialize(model, optimizer, **amp_params)
    if optimizer is not None:
        model, optimizer = amp_result
    else:
        model = amp_result

    # convert model/optimizer back to dict if it needed
    if model_keys is not None:
        model = OrderedDict([(k, v) for k, v in zip(model_keys, model)])
    if optimizer_keys is not None:
        optimizers = [(k, v) for k, v in zip(optimizer_keys, optimizer)]
        optimizer = OrderedDict(optimizers)
    return model, optimizer


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


def get_distributed_mean(value: Union[float, torch.Tensor]):
    """Computes distributed mean among all nodes."""
    if check_torch_distributed_initialized():
        # Fix for runtime warning:
        # To copy construct from a tensor, it is recommended to use
        # sourceTensor.clone().detach() or
        # sourceTensor.clone().detach().requires_grad_(True),
        # rather than torch.tensor(sourceTensor).
        if torch.is_tensor(value):
            value = value.clone().detach().to(device=f"cuda:{torch.cuda.current_device()}")
        else:
            value = torch.tensor(
                value,
                dtype=torch.float,
                device=f"cuda:{torch.cuda.current_device()}",
                requires_grad=False,
            )
        torch.distributed.all_reduce(value)
        value = float(value.item() / torch.distributed.get_world_size())
    return value


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
    local_rank: int, rank: int, world_size: int, use_cuda_visible_devices: bool = True,
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
    "check_apex_available",
    "check_amp_available",
    "check_torch_distributed_initialized",
    "check_slurm_available",
    "assert_fp16_available",
    "initialize_apex",
    "get_nn_from_ddp_module",
    "get_rank",
    "get_distributed_mean",
    "get_distributed_env",
    "get_distributed_params",
    "get_slurm_params",
]
