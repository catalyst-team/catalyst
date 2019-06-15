from typing import Tuple, Dict, Iterable
import os
import copy

import torch
from torch import nn, optim
import torch.backends.cudnn as cudnn

from .ddp import is_wrapped_with_ddp

_Model = nn.Module
_Criterion = nn.Module
_Optimizer = optim.Optimizer
# noinspection PyProtectedMember
_Scheduler = optim.lr_scheduler._LRScheduler


def assert_fp16_available():
    assert torch.backends.cudnn.enabled, \
        "fp16 mode requires cudnn backend to be enabled."

    try:
        __import__('apex')
    except ImportError:
        assert False, \
            "NVidia Apex package must be installed. " \
            "See https://github.com/NVIDIA/apex."


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_optimizable_params(model_or_params):
    params: Iterable[torch.Tensor] = model_or_params
    if isinstance(model_or_params, nn.Module):
        params = model_or_params.parameters()

    master_params = [p for p in params if p.requires_grad]
    return master_params


def process_components(
    model: _Model,
    criterion: _Criterion = None,
    optimizer: _Optimizer = None,
    scheduler: _Scheduler = None,
    distributed_params: Dict = None
) -> Tuple[_Model, _Criterion, _Optimizer, _Scheduler, torch.device]:
    distributed_params = distributed_params or {}
    distributed_params = copy.deepcopy(distributed_params)
    device = get_device()

    if torch.cuda.is_available():
        benchmark = os.environ.get("CUDNN_BENCHMARK", "True") == "True"
        cudnn.benchmark = benchmark

    model = model.to(device)

    if is_wrapped_with_ddp(model):
        pass
    elif len(distributed_params) > 0:
        assert_fp16_available()
        from apex import amp

        distributed_rank = distributed_params.pop("rank", -1)

        if distributed_rank > -1:
            torch.cuda.set_device(distributed_rank)
            torch.distributed.init_process_group(
                backend="nccl", init_method="env://")

        model, optimizer = amp.initialize(
            model, optimizer, **distributed_params)

        if distributed_rank > -1:
            from apex.parallel import DistributedDataParallel
            model = DistributedDataParallel(model)
        elif torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
    elif torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    model = model.to(device)

    return model, criterion, optimizer, scheduler, device


def get_activation_fn(activation: str = None):
    if activation is None or activation.lower() == "none":
        activation_fn = lambda x: x  # noqa: E731
    else:
        activation_fn = torch.nn.__dict__[activation]()
    return activation_fn
