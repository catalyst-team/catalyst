from typing import Any, Dict, Tuple, Union
from collections import OrderedDict
import copy
from functools import partial
import logging

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from catalyst.settings import SETTINGS
from catalyst.typing import Model, Sampler
from catalyst.utils.distributed import get_distributed_params, get_rank
from catalyst.utils.misc import merge_dicts, set_global_seed
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
            models[models_keys], layerwise_params, no_bias_weight_decay, lr_scaling
        )
    elif (
        SETTINGS.hydra_required
        and isinstance(models_keys, (list, tuple, ListConfig))
        or isinstance(models_keys, (list, tuple))
    ):
        model_params = []
        for model_key_el in models_keys:
            model_params_el = process_model_params(
                models[model_key_el], layerwise_params, no_bias_weight_decay, lr_scaling
            )
            model_params.extend(model_params_el)
    else:
        raise ValueError(f"unknown type of models_keys {type(models_keys)}")
    return model_params


def _worker_init_fn(x, *, initial_seed):
    return set_global_seed(initial_seed + x)


def get_loaders_from_params(
    batch_size: int = 1,
    num_workers: int = 0,
    drop_last: bool = False,
    per_gpu_scaling: bool = False,
    loaders_params: Dict[str, Any] = None,
    samplers: "OrderedDict[str, Sampler]" = None,
    datasets: "OrderedDict[str, Union[Dataset, dict]]" = None,
    initial_seed: int = 42,
) -> "OrderedDict[str, DataLoader]":
    """
    Creates pytorch dataloaders from datasets and additional parameters.

    Args:
        batch_size: ``batch_size`` parameter
            from ``torch.utils.data.DataLoader``
        num_workers: ``num_workers`` parameter
            from ``torch.utils.data.DataLoader``
        drop_last: ``drop_last`` parameter
            from ``torch.utils.data.DataLoader``
        per_gpu_scaling: boolean flag,
            if ``True``, scales batch_size in proportion to the number of GPUs
        loaders_params: additional loaders parameters
        samplers: additional sampler parameters
        initial_seed: initial seed for ``torch.utils.data.DataLoader``
            workers
        datasets: ordered dictionary with ``torch.utils.data.Dataset``

    Returns:
        OrderedDict[str, DataLoader]: dictionary with
            ``torch.utils.data.DataLoader``

    Raises:
        NotImplementedError: if datasource is out of ``Dataset`` or dict
        ValueError: if batch_sampler option is mutually
            exclusive with distributed
    """
    from catalyst.data.sampler import DistributedSamplerWrapper

    default_batch_size = batch_size
    default_num_workers = num_workers
    loaders_params = copy.deepcopy(loaders_params) or {}
    assert isinstance(loaders_params, dict), (
        f"`loaders_params` should be a Dict. " f"Got: {loaders_params}"
    )
    samplers = copy.deepcopy(samplers) or {}
    assert isinstance(samplers, dict), f"`samplers` should be a Dict. Got: {samplers}"
    datasets = datasets if datasets is not None else {}

    distributed_rank = get_rank()
    distributed = distributed_rank > -1

    loaders = OrderedDict()
    for name, datasource in datasets.items():
        assert isinstance(
            datasource, (Dataset, dict)
        ), f"{datasource} should be Dataset or Dict. Got: {datasource}"

        loader_params = loaders_params.pop(name, {})
        assert isinstance(loader_params, dict), f"{loader_params} should be Dict"

        sampler: Sampler = None
        if isinstance(datasource, dict) and "sampler" in datasource:
            sampler = datasource.pop("sampler", None)
        sampler = samplers.pop(name, sampler)

        batch_size = loader_params.pop("batch_size", default_batch_size)
        num_workers = loader_params.pop("num_workers", default_num_workers)

        if per_gpu_scaling and not distributed:
            num_gpus = max(1, torch.cuda.device_count())
            batch_size *= num_gpus
            num_workers *= num_gpus
        elif not per_gpu_scaling and distributed:
            world_size = get_distributed_params().pop("world_size", 1)
            if batch_size % world_size == 0:
                batch_size = int(batch_size / world_size)
            else:
                raise ValueError(
                    "For this distributed mode with per_gpu_scaling = False "
                    "you need to have batch_size divisible by number of GPUs"
                )

        loader_params = {
            "batch_size": batch_size,
            "num_workers": num_workers,
            "pin_memory": torch.cuda.is_available(),
            "drop_last": drop_last,
            **loader_params,
        }

        if isinstance(datasource, Dataset):
            loader_params["dataset"] = datasource
        elif isinstance(datasource, dict):
            assert "dataset" in datasource, "You need to specify dataset for dataloader"
            loader_params = merge_dicts(datasource, loader_params)
        else:
            raise NotImplementedError

        if distributed:
            if sampler is not None:
                if not isinstance(sampler, DistributedSampler):
                    sampler = DistributedSamplerWrapper(sampler=sampler)
            else:
                sampler = DistributedSampler(dataset=loader_params["dataset"])

        loader_params["shuffle"] = name.startswith("train") and sampler is None

        loader_params["sampler"] = sampler

        if "batch_sampler" in loader_params:
            if distributed:
                raise ValueError("batch_sampler option is mutually " "exclusive with distributed")

            for k in ("batch_size", "shuffle", "sampler", "drop_last"):
                loader_params.pop(k, None)

        if "worker_init_fn" not in loader_params:
            loader_params["worker_init_fn"] = partial(_worker_init_fn, initial_seed=initial_seed)

        loaders[name] = DataLoader(**loader_params)

    return loaders


__all__ = ["do_lr_linear_scaling", "get_model_parameters", "get_loaders_from_params"]
