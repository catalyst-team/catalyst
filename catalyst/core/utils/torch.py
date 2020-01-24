from typing import Dict, Tuple  # isort:skip
import copy

import torch
from torch import nn

from catalyst.core import utils
from catalyst.utils.typing import (
    Criterion, Device, Model, Optimizer, Scheduler
)


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
    if device is None:
        device = utils.get_device()

    model.to(device=device)

    if utils.is_wrapped_with_ddp(model):
        pass
    elif len(distributed_params) > 0:
        assert isinstance(model, nn.Module)
        distributed_rank = distributed_params.pop("rank", -1)
        syncbn = distributed_params.pop("syncbn", False)

        if distributed_rank > -1:
            torch.cuda.set_device(distributed_rank)
            torch.distributed.init_process_group(
                backend="nccl", init_method="env://"
            )

        if "opt_level" in distributed_params:
            utils.assert_fp16_available()
            from apex import amp

            amp_result = amp.initialize(
                model, optimizer, **distributed_params
            )
            if optimizer is not None:
                model, optimizer = amp_result
            else:
                model = amp_result

            if distributed_rank > -1:
                from apex.parallel import DistributedDataParallel
                model = DistributedDataParallel(model)

                if syncbn:
                    from apex.parallel import convert_syncbn_model
                    model = convert_syncbn_model(model)

        if distributed_rank <= -1 and torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
    elif torch.cuda.device_count() > 1:
        if isinstance(model, nn.Module):
            model = torch.nn.DataParallel(model)
        elif isinstance(model, dict):
            model = {k: torch.nn.DataParallel(v) for k, v in model.items()}

    model.to(device=device)

    return model, criterion, optimizer, scheduler, device


__all__ = ["process_components"]
