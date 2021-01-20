from typing import Any, Dict, Tuple, Union
import logging

import torch
from torch import nn

from catalyst.callbacks.batch_overfit import BatchOverfitCallback
from catalyst.callbacks.checkpoint import CheckpointCallback
from catalyst.callbacks.criterion import CriterionCallback
from catalyst.callbacks.early_stop import CheckRunCallback
from catalyst.callbacks.exception import ExceptionCallback
from catalyst.callbacks.logging import ConsoleLogger, TensorboardLogger, VerboseLogger
from catalyst.callbacks.metric import MetricManagerCallback
from catalyst.callbacks.optimizer import IOptimizerCallback, OptimizerCallback
from catalyst.callbacks.scheduler import ISchedulerCallback, SchedulerCallback
from catalyst.callbacks.timer import TimerCallback
from catalyst.callbacks.validation import ValidationManagerCallback
from catalyst.core.callback import Callback
from catalyst.core.functional import check_callback_isinstance
from catalyst.settings import IS_HYDRA_AVAILABLE
from catalyst.typing import Model, Optimizer
from catalyst.utils.checkpoint import load_checkpoint, unpack_checkpoint
from catalyst.utils.distributed import get_rank
from catalyst.utils.torch import any2device, get_device, process_model_params

if IS_HYDRA_AVAILABLE:
    from omegaconf import ListConfig

logger = logging.getLogger(__name__)


def process_callbacks(callbacks: Dict[str, Callback], stage_index: int = None) -> None:
    """
    Iterate over each of the callbacks and update
    appropriate parameters required for success
    run of config experiment.

    Arguments:
        callbacks (Dict[str, Callback]): finalized order of callbacks.
        stage_index (int): number of a current stage

    """
    if stage_index is None:
        stage_index = -float("inf")
    for callback in callbacks.values():
        # NOTE: in experiments with multiple stages need to omit
        #       loading of a best model state for the first stage
        #       but for the other stages by default should
        #       load best state of a model
        # @TODO: move this logic to ``CheckpointCallback``
        if isinstance(callback, CheckpointCallback) and stage_index > 0:
            if callback.load_on_stage_start is None:
                callback.load_on_stage_start = "best"
            if (
                isinstance(callback.load_on_stage_start, dict)
                and "model" not in callback.load_on_stage_start
            ):
                callback.load_on_stage_start["model"] = "best"


def add_default_callbacks(
    callbacks,
    verbose: bool,
    check_time: bool,
    check_run: bool,
    overfit: bool,
    is_infer: bool,
    is_logger: bool = False,
    is_criterion: bool = False,
    is_optimizer: bool = False,
    is_scheduler: bool = False,
):
    """
    Adds to user callbacks default ones due to user flags and config

    Args:
        callbacks: user callbacks
        verbose: verbose config flag
        check_time: check time config flag
        check_run: check run config flag
        overfit: overfit config flag
        is_infer: is stage is infer stage
        is_logger: is there logdir
        is_criterion: is there criterion
        is_optimizer: is there optimizer
        is_scheduler: is there scheduler

    Returns:
        user callbacks + default callbacks

    """
    default_callbacks = []

    optimizer_cls = OptimizerCallback

    if verbose:
        default_callbacks.append(("_verbose", None, VerboseLogger))
    if check_time:
        default_callbacks.append(("_timer", None, TimerCallback))
    if check_run:
        default_callbacks.append(("_check", None, CheckRunCallback))
    if overfit:
        default_callbacks.append(("_overfit", None, BatchOverfitCallback))

    if not is_infer:
        default_callbacks.append(("_metrics", None, MetricManagerCallback))
        default_callbacks.append(("_validation", None, ValidationManagerCallback))
        default_callbacks.append(("_console", None, ConsoleLogger))

        if is_logger:
            default_callbacks.append(("_saver", None, CheckpointCallback))
            default_callbacks.append(("_tensorboard", None, TensorboardLogger))

        if is_criterion:
            default_callbacks.append(("_criterion", None, CriterionCallback))
        if is_optimizer:
            default_callbacks.append(("_optimizer", IOptimizerCallback, optimizer_cls))
        if is_scheduler:
            default_callbacks.append(("_scheduler", ISchedulerCallback, SchedulerCallback))

    default_callbacks.append(("_exception", None, ExceptionCallback))

    for (callback_name, callback_interface, callback_fn) in default_callbacks:
        callback_interface = callback_interface or callback_fn
        is_already_present = any(
            check_callback_isinstance(x, callback_interface) for x in callbacks.values()
        )
        if not is_already_present:
            callbacks[callback_name] = callback_fn()

    return callbacks


def load_optimizer_from_checkpoint(
    optimizer: Optimizer,
    checkpoint_path: str,
    checkpoint_optimizer_key: str,
    model_parameters,
    optimizer_params,
) -> Optimizer:
    """
    Loads optimizer state from checkpoint

    Args:
        optimizer: optimizer
        checkpoint_path: path to checkpoint file
        checkpoint_optimizer_key: key if optimizer checkpoint
                                  in checkpoint state dict
        model_parameters: model parameters
        optimizer_params: optimizer config parameters

    Returns:
        optimizer loaded from checkpoint

    """
    checkpoint = load_checkpoint(checkpoint_path)
    dict2load = optimizer
    if checkpoint_optimizer_key is not None:
        dict2load = {checkpoint_optimizer_key: optimizer}
    unpack_checkpoint(checkpoint, optimizer=dict2load)
    # move optimizer to device
    device = get_device()
    for param in model_parameters:
        param = param["params"][0]
        optimizer_state = optimizer.state[param]
        for state_key, state_value in optimizer_state.items():
            optimizer_state[state_key] = any2device(state_value, device)
    # update optimizer params
    for key, value in optimizer_params.items():
        for optimizer_param_group in optimizer.param_groups:
            optimizer_param_group[key] = value

    return optimizer


def do_lr_linear_scaling(
    lr_scaling_params, batch_size: int, per_gpu_scaling: bool
) -> Tuple[float, float]:
    """
    Linear scaling rule from https://arxiv.org/pdf/1706.02677.pdf

    Args:
        lr_scaling_params: config parameters of lr linear scaling
        batch_size: batch size
        per_gpu_scaling: per-gpu-scaling flag

    Returns:
        lr, lr_scaling

    """
    distributed_rank = get_rank()
    distributed = distributed_rank > -1
    if per_gpu_scaling and not distributed:
        num_gpus = max(1, torch.cuda.device_count())
        batch_size *= num_gpus

    base_lr = lr_scaling_params.get("lr")
    base_batch_size = lr_scaling_params.get("base_batch_size", 256)
    lr_scaling = batch_size / base_batch_size
    lr = base_lr * lr_scaling  # scale default lr
    return lr, lr_scaling


def get_model_parameters(
    models: Union[Model, Dict[str, Model]],
    models_keys: Any,
    layerwise_params: Dict[str, dict],
    no_bias_weight_decay: bool,
    lr_scaling: float,
):  # noqa: DAR401
    """
    Model parameters getter

    Args:
        models: model or dict of models
        models_keys: models keys for which parameters is needed
        layerwise_params: layerwise config parameters
        no_bias_weight_decay: no-bias-weight-decay flag
        lr_scaling: lr scaling number

    Returns:
        models parameters

    """
    if models_keys is None:
        assert isinstance(
            models, nn.Module
        ), "model is key-value, but optimizer has no specified model"
        model_params = process_model_params(
            models, layerwise_params, no_bias_weight_decay, lr_scaling
        )
    elif isinstance(models_keys, str):
        model_params = process_model_params(
            models[models_keys], layerwise_params, no_bias_weight_decay, lr_scaling,
        )
    elif (
        IS_HYDRA_AVAILABLE
        and isinstance(models_keys, (list, tuple, ListConfig))
        or isinstance(models_keys, (list, tuple))
    ):
        model_params = []
        for model_key_el in models_keys:
            model_params_el = process_model_params(
                models[model_key_el], layerwise_params, no_bias_weight_decay, lr_scaling,
            )
            model_params.extend(model_params_el)
    else:
        raise ValueError(f"unknown type of models_keys {type(models_keys)}")
    return model_params


__all__ = [
    "process_callbacks",
    "add_default_callbacks",
    "load_optimizer_from_checkpoint",
    "do_lr_linear_scaling",
    "get_model_parameters",
]
