from typing import Any, Callable, Dict, Iterable
from collections import OrderedDict
import copy
from functools import partial

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

    dataset = ListDataset(list_data=data_source, open_fn=open_fn, dict_transform=dict_transform)
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


def _worker_init_fn(x, *, initial_seed):
    return set_global_seed(initial_seed + x)


def get_loaders_from_params(
    batch_size: int = 1,
    num_workers: int = 0,
    drop_last: bool = False,
    per_gpu_scaling: bool = False,
    loaders_params: Dict[str, Any] = None,
    samplers_params: Dict[str, Any] = None,
    initial_seed: int = 42,
    datasets_fn: Callable = None,
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
        datasets_fn(Callable): callable function to get dictionary with
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
    loaders_params = copy.deepcopy(loaders_params) or {}
    assert isinstance(loaders_params, dict), (
        f"`loaders_params` should be a Dict. " f"Got: {loaders_params}"
    )
    samplers_params = copy.deepcopy(samplers_params) or {}
    assert isinstance(
        samplers_params, dict
    ), f"`samplers_params` should be a Dict. Got: {samplers_params}"

    distributed_rank = get_rank()
    distributed = distributed_rank > -1

    if datasets_fn is not None:
        datasets = datasets_fn(**data_params)
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
            loader_params["worker_init_fn"] = partial(_worker_init_fn, initial_seed=initial_seed)

        loaders[name] = DataLoader(**loader_params)

    return loaders


__all__ = [
    "get_loader",
    "get_loaders_from_params",
]
