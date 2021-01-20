from typing import Any, Callable, Dict, Iterable, Union
from collections import OrderedDict
from copy import copy
import warnings

import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torch.utils.data.dataloader import default_collate as default_collate_fn

from catalyst.registry import REGISTRY
from catalyst.utils.distributed import get_distributed_params, get_rank
from catalyst.utils.misc import merge_dicts, set_global_seed


def get_loader(
    data_source: Iterable[dict],
    open_fn: Callable,
    dict_transform: Callable = None,
    sampler=None,
    collate_fn: Callable = default_collate_fn,
    batch_size: int = 32,
    num_workers: int = 4,
    shuffle: bool = False,
    drop_last: bool = False,
):
    """Creates a DataLoader from given source and its open/transform params.

    Args:
        data_source: and iterable containing your
            data annotations,
            (for example path to images, labels, bboxes, etc)
        open_fn: function, that can open your
            annotations dict and
            transfer it to data, needed by your network
            (for example open image by path, or tokenize read string)
        dict_transform: transforms to use on dict
            (for example normalize image, add blur, crop/resize/etc)
        sampler (Sampler, optional): defines the strategy to draw samples from
            the dataset
        collate_fn (callable, optional): merges a list of samples to form a
            mini-batch of Tensor(s).  Used when using batched loading from a
            map-style dataset
        batch_size (int, optional): how many samples per batch to load
        num_workers (int, optional): how many subprocesses to use for data
            loading. ``0`` means that the data will be loaded
            in the main process
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: ``False``).
        drop_last (bool, optional): set to ``True`` to drop
            the last incomplete batch, if the dataset size is not divisible
            by the batch size. If ``False`` and the size of dataset
            is not divisible by the batch size, then the last batch
            will be smaller. (default: ``False``)

    Returns:
        DataLoader with ``catalyst.data.ListDataset``
    """
    from catalyst.data.dataset import ListDataset

    dataset = ListDataset(list_data=data_source, open_fn=open_fn, dict_transform=dict_transform,)
    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        sampler=sampler,
        collate_fn=collate_fn,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=torch.cuda.is_available(),
        drop_last=drop_last,
    )
    return loader


def get_native_batch_from_loader(loader: DataLoader, batch_index: int = 0):
    """
    Returns a batch from experiment loader

    Args:
        loader: Loader to get batch from
        batch_index: Index of batch to take from dataset of the loader

    Returns:
        batch from loader
    """
    dataset = loader.dataset
    collate_fn = loader.collate_fn
    return collate_fn([dataset[batch_index]])


def get_native_batch_from_loaders(
    loaders: Dict[str, DataLoader], loader: Union[str, int] = 0, batch_index: int = 0,
):
    """
    Returns a batch from experiment loaders by its index or name.

    Args:
        loaders (Dict[str, DataLoader]): Loaders list to get loader from
        loader (Union[str, int]): Loader name or its index, default is zero
        batch_index: Index of batch to take from dataset of the loader

    Returns:
        batch from loader

    Raises:
        TypeError: if loader parameter is not a string or an integer
    """
    if isinstance(loader, str):
        loader_instance = loaders[loader]
    elif isinstance(loader, int):
        loader_instance = list(loaders.values())[loader]
    else:
        raise TypeError("Loader parameter must be a string or an integer")

    output = get_native_batch_from_loader(loader=loader_instance, batch_index=batch_index)

    return output


def _force_make_distributed_loader(loader: DataLoader) -> DataLoader:
    """
    Transfers loader to distributed mode. Experimental feature.

    Args:
        loader: pytorch dataloder

    Returns:
        DataLoader: pytorch dataloder with distributed sampler.
    """
    from catalyst.data.sampler import DistributedSamplerWrapper

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
    from catalyst.data.sampler import DistributedSamplerWrapper

    rank = get_rank()
    if rank >= 0:
        for key, value in loaders.items():
            if not isinstance(value.sampler, (DistributedSampler, DistributedSamplerWrapper)):
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
        batch_size: ``batch_size`` parameter
            from ``torch.utils.data.DataLoader``
        num_workers: ``num_workers`` parameter
            from ``torch.utils.data.DataLoader``
        drop_last: ``drop_last`` parameter
            from ``torch.utils.data.DataLoader``
        per_gpu_scaling: boolean flag,
            if ``True``, scales batch_size in proportion to the number of GPUs
        loaders_params (Dict[str, Any]): additional loaders parameters
        samplers_params (Dict[str, Any]): additional sampler parameters
        initial_seed: initial seed for ``torch.utils.data.DataLoader``
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
    from catalyst.data.sampler import DistributedSamplerWrapper

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
        assert isinstance(loader_params, dict), f"{loader_params} should be Dict"

        sampler_params = samplers_params.pop(name, None)
        if sampler_params is None:
            if isinstance(datasource, dict) and "sampler" in datasource:
                sampler = datasource.pop("sampler", None)
            else:
                sampler = None
        else:
            sampler = REGISTRY.get_from_params(**sampler_params)
            if isinstance(datasource, dict) and "sampler" in datasource:
                datasource.pop("sampler", None)

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
            loader_params["worker_init_fn"] = lambda x: set_global_seed(initial_seed + x)

        loaders[name] = DataLoader(**loader_params)

    return loaders


__all__ = [
    "get_native_batch_from_loader",
    "get_native_batch_from_loaders",
    "get_loader",
    "validate_loaders",
    "get_loaders_from_params",
    "validate_loaders",
]
