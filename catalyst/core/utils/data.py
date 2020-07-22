from typing import Any, Callable, Dict
from collections import OrderedDict
from copy import copy
import warnings

import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from catalyst.data import DistributedSamplerWrapper
from catalyst.registry import SAMPLER
from catalyst.utils import get_rank, merge_dicts, set_global_seed


def _force_make_distributed_loader(loader: DataLoader) -> DataLoader:
    """
    Transfers loader to distributed mode. Experimental feature.

    Args:
        loader (DataLoader): pytorch dataloder

    Returns:
        DataLoader: pytorch dataloder with distributed sampler.
    """
    sampler = (
        DistributedSampler(dataset=loader.dataset)
        if getattr(loader, "sampler", None) is not None
        else DistributedSamplerWrapper(sampler=loader.sampler)
    )
    loader = DataLoader(
        dataset=copy(loader.dataset),
        batch_size=loader.batch_size,
        # shuffle=loader.shuffle,
        sampler=sampler,
        # batch_sampler=loader.batch_sampler,
        num_workers=loader.num_workers,
        # collate_fn=loader.collate_fn,
        pin_memory=loader.pin_memory,
        drop_last=loader.drop_last,
    )
    return loader


def validate_loaders(loaders: Dict[str, DataLoader]) -> Dict[str, DataLoader]:
    """
    Check pytorch dataloaders for distributed setup.
    Transfers them to distirbuted mode if necessary.
    (Experimental feature)

    Args:
        loaders (Dict[str, DataLoader]): dictionery with pytorch dataloaders

    Returns:
        Dict[str, DataLoader]: dictionery
            with pytorch dataloaders (with distributed samplers if necessary)
    """
    rank = get_rank()
    if rank >= 0:
        for key, value in loaders.items():
            if not isinstance(
                value.sampler, (DistributedSampler, DistributedSamplerWrapper)
            ):
                warnings.warn(
                    "With distributed training setup, "
                    "you need ``DistributedSampler`` for your ``DataLoader``."
                    "Transferring to distributed mode. (Experimental feature)"
                )
                loaders[key] = _force_make_distributed_loader(value)
    return loaders


def get_loaders_from_params(
    batch_size: int = 1,
    num_workers: int = 0,
    drop_last: bool = False,
    per_gpu_scaling: bool = False,
    loaders_params: Dict[str, Any] = None,
    samplers_params: Dict[str, Any] = None,
    initial_seed: int = 42,
    get_datasets_fn: Callable = None,
    **data_params,
) -> "OrderedDict[str, DataLoader]":
    """
    Creates pytorch dataloaders from datasets and additional parameters.

    Args:
        batch_size (int): ``batch_size`` parameter
            from ``torch.utils.data.DataLoader``
        num_workers (int): ``num_workers`` parameter
            from ``torch.utils.data.DataLoader``
        drop_last (bool): ``drop_last`` parameter
            from ``torch.utils.data.DataLoader``
        per_gpu_scaling (bool): boolean flag,
            if ``True``, uses ``batch_size=batch_size*num_available_gpus``
        loaders_params (Dict[str, Any]): additional loaders parameters
        samplers_params (Dict[str, Any]): additional sampler parameters
        initial_seed (int): initial seed for ``torch.utils.data.DataLoader``
            workers
        get_datasets_fn(Callable): callable function to get dictionary with
            ``torch.utils.data.Datasets``
        **data_params: additional data parameters
            or dictionary with ``torch.utils.data.Datasets`` to use for
            pytorch dataloaders creation

    Returns:
        OrderedDict[str, DataLoader]: dictionary with
            ``torch.utils.data.DataLoader``

    Raises:
        NotImplementedError: if datasource is out of `Dataset` or dict
        ValueError: if batch_sampler option is mutually
            exclusive with distributed
    """
    default_batch_size = batch_size
    default_num_workers = num_workers
    loaders_params = loaders_params or {}
    assert isinstance(loaders_params, dict), (
        f"`loaders_params` should be a Dict. " f"Got: {loaders_params}"
    )
    samplers_params = samplers_params or {}
    assert isinstance(
        samplers_params, dict
    ), f"`samplers_params` should be a Dict. Got: {samplers_params}"

    distributed_rank = get_rank()
    distributed = distributed_rank > -1

    if get_datasets_fn is not None:
        datasets = get_datasets_fn(**data_params)
    else:
        datasets = dict(**data_params)

    loaders = OrderedDict()
    for name, datasource in datasets.items():  # noqa: WPS426
        assert isinstance(
            datasource, (Dataset, dict)
        ), f"{datasource} should be Dataset or Dict. Got: {datasource}"

        loader_params = loaders_params.pop(name, {})
        assert isinstance(
            loader_params, dict
        ), f"{loader_params} should be Dict"

        sampler_params = samplers_params.pop(name, None)
        if sampler_params is None:
            if isinstance(datasource, dict) and "sampler" in datasource:
                sampler = datasource.pop("sampler", None)
            else:
                sampler = None
        else:
            sampler = SAMPLER.get_from_params(**sampler_params)
            if isinstance(datasource, dict) and "sampler" in datasource:
                datasource.pop("sampler", None)

        batch_size = loader_params.pop("batch_size", default_batch_size)
        num_workers = loader_params.pop("num_workers", default_num_workers)

        if per_gpu_scaling and not distributed:
            num_gpus = max(1, torch.cuda.device_count())
            batch_size *= num_gpus
            num_workers *= num_gpus

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
            assert (
                "dataset" in datasource
            ), "You need to specify dataset for dataloader"
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
                raise ValueError(
                    "batch_sampler option is mutually "
                    "exclusive with distributed"
                )

            for k in ("batch_size", "shuffle", "sampler", "drop_last"):
                loader_params.pop(k, None)

        if "worker_init_fn" not in loader_params:
            loader_params["worker_init_fn"] = lambda x: set_global_seed(
                initial_seed + x
            )

        loaders[name] = DataLoader(**loader_params)

    return loaders


__all__ = ["get_loaders_from_params", "validate_loaders"]
