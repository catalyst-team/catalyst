from typing import Dict, Tuple
import copy

import torch
from torch import nn
import torch.distributed

from catalyst.settings import IS_XLA_AVAILABLE
from catalyst.typing import Criterion, Device, Model, Optimizer, RunnerModel, Scheduler
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


def _patch_forward(model):
    import apex

    input_caster_lambda = (
        lambda tensor: tensor.to(
            apex.amp._amp_state.opt_properties.options["cast_model_type"]  # noqa: WPS437
        )
        if tensor.is_floating_point()
        else tensor
    )
    output_caster_lambda = (
        lambda tensor: tensor.to(
            apex.amp._amp_state.opt_properties.options.get(  # noqa: WPS437
                "cast_model_outputs", torch.float32
            )
        )
        if tensor.is_floating_point()
        else tensor
    )

    def new_fwd(
        *args,
        old_fwd=model.forward,
        input_caster=input_caster_lambda,
        output_caster=output_caster_lambda,
        **kwargs,
    ):
        return apex.amp._initialize.applier(  # noqa: WPS437
            old_fwd(
                *apex.amp._initialize.applier(args, input_caster),  # noqa: WPS437
                **apex.amp._initialize.applier(kwargs, input_caster),  # noqa: WPS437
            ),
            output_caster,
        )

    model.forward = new_fwd
    return model


# apex issue https://github.com/deepset-ai/FARM/issues/210
# solution: https://github.com/NVIDIA/apex/issues/503#issuecomment-566181771
def _wrap_into_data_parallel_with_apex(
    model: RunnerModel, optimizer: Optimizer, engine_params: Dict
):
    if isinstance(model, nn.Module):
        model = nn.Sequential(model)
        model, optimizer = initialize_apex(model, optimizer, **engine_params)
        model = torch.nn.DataParallel(model[0])
        model = _patch_forward(model)
    elif isinstance(model, dict):
        model = {k: nn.Sequential(v) for k, v in model.items()}
        model, optimizer = initialize_apex(model, optimizer, **engine_params)
        model = {k: nn.DataParallel(v[0]) for k, v in model.items()}
        model = {k: _patch_forward(v) for k, v in model.items()}
    else:
        raise NotImplementedError()

    return model, optimizer


def process_components(
    model: RunnerModel,
    criterion: Criterion = None,
    optimizer: Optimizer = None,
    scheduler: Scheduler = None,
    engine_params: Dict = None,
    device: Device = None,
) -> Tuple[RunnerModel, Criterion, Optimizer, Scheduler, Device]:
    """
    Returns the processed model, criterion, optimizer, scheduler and device.

    Args:
        model: torch model
        criterion: criterion function
        optimizer: optimizer
        scheduler: scheduler
        engine_params (dict, optional): dict with the parameters
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
    engine_params = engine_params or {}
    engine_params = copy.deepcopy(engine_params)
    engine_params.update(get_distributed_params())

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

    is_apex_enabled = engine_params.get("apex", False) and check_apex_available()
    is_amp_enabled = engine_params.get("amp", False) and check_amp_available()

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
        assert isinstance(model, nn.Module), "Distributed training is not available for KV model"

        local_rank = engine_params.pop("local_rank", 0) or 0
        device = f"cuda:{local_rank}"
        model = maybe_recursive_call(model, "to", device=device)

        syncbn = engine_params.pop("syncbn", False)

        if is_apex_enabled:
            import apex

            if syncbn:
                model = apex.parallel.convert_syncbn_model(model)

            model, optimizer = initialize_apex(model, optimizer, **engine_params)
            model = apex.parallel.DistributedDataParallel(model)
        else:
            if syncbn:
                model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

            model = nn.parallel.DistributedDataParallel(
                model, device_ids=[local_rank], output_device=local_rank
            )
    # data parallel run (dp) (with apex support)
    else:
        is_data_parallel = (
            torch.cuda.device_count() > 1 and device.type != "cpu" and device.index is None
        )

        if is_apex_enabled and not is_data_parallel:
            model, optimizer = initialize_apex(model, optimizer, **engine_params)

        elif not is_apex_enabled and is_data_parallel:
            if isinstance(model, nn.Module):
                model = nn.DataParallel(model)
            elif isinstance(model, dict):
                model = {k: nn.DataParallel(v) for k, v in model.items()}
            else:
                raise NotImplementedError()

        elif is_apex_enabled and is_data_parallel:
            model, optimizer = _wrap_into_data_parallel_with_apex(model, optimizer, engine_params)

    model: Model = maybe_recursive_call(model, "to", device=device)

    return model, criterion, optimizer, scheduler, device


__all__ = ["process_components"]
