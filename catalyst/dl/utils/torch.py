from typing import Dict, Tuple, Iterable, Callable  # isort:skip
import copy

import torch
from torch import nn
from torch.utils.data.dataloader import default_collate as default_collate_fn

from catalyst.dl import utils
from catalyst.utils import maybe_recursive_call
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

    model: Model = maybe_recursive_call(model, "to", device=device)

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

    model = maybe_recursive_call(model, "to", device=device)

    return model, criterion, optimizer, scheduler, device


def get_loader(
    data_source: Iterable[dict],
    open_fn: Callable,
    dict_transform: Callable = None,
    sampler=None,
    collate_fn: Callable = default_collate_fn,
    batch_size: int = 32,
    num_workers: int = 4,
    shuffle: bool = False,
    drop_last: bool = False
):
    """
    Creates a DataLoader from given source and its open/transform params

    Args:
        data_source (Iterable[dict]): and iterable containing your
            data annotations,
            (for example path to images, labels, bboxes, etc)
        open_fn (Callable): function, that can open your
            annotations dict and
            transfer it to data, needed by your network
            (for example open image by path, or tokenize read string)
        dict_transform (callable): transforms to use on dict
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
    from catalyst.data import ListDataset

    dataset = ListDataset(
        list_data=data_source,
        open_fn=open_fn,
        dict_transform=dict_transform,
    )
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


__all__ = [
    "process_components",
    "get_loader"
]
