from typing import Dict, Tuple
import copy

import torch
from torch import nn
import torch.distributed

from catalyst.tools.settings import IS_XLA_AVAILABLE
from catalyst.tools.typing import (
    Criterion,
    Device,
    Model,
    Optimizer,
    Scheduler,
)
from catalyst.utils.distributed import (
    check_amp_available,
    check_apex_available,
    check_ddp_wrapped,
    get_distributed_params,
    get_rank,
    initialize_apex,
)
from catalyst.utils.misc import maybe_recursive_call
from catalyst.utils.torch import get_device


def process_components(
    model: Model,
    criterion: Criterion = None,
    optimizer: Optimizer = None,
    scheduler: Scheduler = None,
    distributed_params: Dict = None,
    device: Device = None,
) -> Tuple[Model, Criterion, Optimizer, Scheduler, Device]:
    """
    Returns the processed model, criterion, optimizer, scheduler and device.

    Args:
        model (Model): torch model
        criterion (Criterion): criterion function
        optimizer (Optimizer): optimizer
        scheduler (Scheduler): scheduler
        distributed_params (dict, optional): dict with the parameters
            for distributed and FP16 method
        device (Device, optional): device

    Returns:
        tuple with processed model, criterion, optimizer, scheduler and device.

    Raises:
        ValueError: if device is None and TPU available,
            for using TPU need to manualy move model/optimizer/scheduler
            to a TPU device and pass device to a function.
        NotImplementedError: if model is not nn.Module or dict for multi-gpu,
            nn.ModuleDict for DataParallel not implemented yet
    """
    distributed_params = distributed_params or {}
    distributed_params = copy.deepcopy(distributed_params)
    distributed_params.update(get_distributed_params())

    if device is None and IS_XLA_AVAILABLE:
        raise ValueError(
            "TPU device is available. "
            "Please move model, optimizer and scheduler (if present) "
            "to TPU device manualy and specify a device or "
            "use CPU device."
        )

    if device is None:
        device = get_device()
    elif isinstance(device, str):
        device = torch.device(device)

    is_apex_enabled = (
        distributed_params.pop("apex", False) and check_apex_available()
    )

    is_amp_enabled = (
        distributed_params.get("amp", False) and check_amp_available()
    )

    if is_apex_enabled and is_amp_enabled:
        raise ValueError(
            "Both NVidia Apex and Torch.Amp are enabled. "
            "You must choose only one mixed precision backend"
        )
    model: Model = maybe_recursive_call(model, "to", device=device)

    if check_ddp_wrapped(model):
        pass
    # distributed data parallel run (ddp) (with apex support)
    elif get_rank() >= 0:
        assert isinstance(
            model, nn.Module
        ), "Distributed training is not available for KV model"

        local_rank = distributed_params.pop("local_rank", 0) or 0
        device = f"cuda:{local_rank}"
        model = maybe_recursive_call(model, "to", device=device)

        syncbn = distributed_params.pop("syncbn", False)

        if is_apex_enabled:
            import apex

            if syncbn:
                model = apex.parallel.convert_syncbn_model(model)

            model, optimizer = initialize_apex(
                model, optimizer, **distributed_params
            )
            model = apex.parallel.DistributedDataParallel(model)
        else:
            if syncbn:
                model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

            model = nn.parallel.DistributedDataParallel(
                model, device_ids=[local_rank], output_device=local_rank
            )
    # data parallel run (dp) (with apex support)
    else:
        # apex issue https://github.com/deepset-ai/FARM/issues/210
        use_apex = (is_apex_enabled and torch.cuda.device_count() == 1) or (
            is_apex_enabled
            and torch.cuda.device_count() > 1
            and distributed_params.get("opt_level", "O0") == "O1"
        )

        if use_apex:
            assert isinstance(
                model, nn.Module
            ), "Apex training is not available for KV model"

            model, optimizer = initialize_apex(
                model, optimizer, **distributed_params
            )

        if (
            torch.cuda.device_count() > 1
            and device.type != "cpu"
            and device.index is None
        ):
            if isinstance(model, nn.Module):
                model = nn.DataParallel(model)
            elif isinstance(model, dict):
                model = {k: nn.DataParallel(v) for k, v in model.items()}
            else:
                raise NotImplementedError()

    model: Model = maybe_recursive_call(model, "to", device=device)

    return model, criterion, optimizer, scheduler, device


__all__ = ["process_components"]
