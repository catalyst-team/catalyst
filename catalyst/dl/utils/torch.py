from typing import Tuple, Dict
import os
import copy

import torch
from torch import nn, optim
import torch.backends.cudnn as cudnn
from catalyst.dl import utils

_Model = nn.Module
_Criterion = nn.Module
_Optimizer = optim.Optimizer
# noinspection PyProtectedMember
_Scheduler = optim.lr_scheduler._LRScheduler


def process_components(
    model: _Model,
    criterion: _Criterion = None,
    optimizer: _Optimizer = None,
    scheduler: _Scheduler = None,
    distributed_params: Dict = None
) -> Tuple[_Model, _Criterion, _Optimizer, _Scheduler, torch.device]:
    distributed_params = distributed_params or {}
    distributed_params = copy.deepcopy(distributed_params)
    device = utils.get_device()

    if torch.cuda.is_available():
        benchmark = os.environ.get("CUDNN_BENCHMARK", "True") == "True"
        cudnn.benchmark = benchmark

    model = model.to(device)

    if utils.is_wrapped_with_ddp(model):
        pass
    elif len(distributed_params) > 0:
        utils.assert_fp16_available()
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
