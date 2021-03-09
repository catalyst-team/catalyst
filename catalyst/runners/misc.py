from typing import Any, Dict, Tuple, Union
import logging

import torch
from torch import nn

from catalyst.settings import SETTINGS
from catalyst.typing import Model
from catalyst.utils.distributed import get_rank
from catalyst.utils.torch import process_model_params

if SETTINGS.hydra_required:
    from omegaconf import ListConfig

logger = logging.getLogger(__name__)


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
        SETTINGS.hydra_required
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


__all__ = ["do_lr_linear_scaling", "get_model_parameters"]
