from typing import Tuple, Dict
import copy

import torch
from torch import nn, optim
from torch.utils.data.dataloader import default_collate as default_collate_fn

from catalyst.dl import utils
from catalyst.utils import maybe_recursive_call

_Model = nn.Module
_Criterion = nn.Module
_Optimizer = optim.Optimizer
# noinspection PyProtectedMember
_Scheduler = optim.lr_scheduler._LRScheduler


def process_components(
    model: _Model,
    criterion: _Criterion = None,
    optimizer: _Optimizer = None,
    scheduler: _Scheduler = None,
    distributed_params: Dict = None
) -> Tuple[_Model, _Criterion, _Optimizer, _Scheduler, torch.device]:
    distributed_params = distributed_params or {}
    distributed_params = copy.deepcopy(distributed_params)
    device = utils.get_device()

    model = maybe_recursive_call(model, "to", device=device)

    if utils.is_wrapped_with_ddp(model):
        pass
    elif len(distributed_params) > 0:
        assert isinstance(model, nn.Module)
        utils.assert_fp16_available()
        from apex import amp
        from apex.parallel import convert_syncbn_model

        distributed_rank = distributed_params.pop("rank", -1)
        syncbn = distributed_params.pop("syncbn", False)

        if distributed_rank > -1:
            torch.cuda.set_device(distributed_rank)
            torch.distributed.init_process_group(
                backend="nccl", init_method="env://"
            )

        model, optimizer = amp.initialize(
            model, optimizer, **distributed_params
        )

        if distributed_rank > -1:
            from apex.parallel import DistributedDataParallel
            model = DistributedDataParallel(model)

            if syncbn:
                model = convert_syncbn_model(model)
        elif torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
    elif torch.cuda.device_count() > 1:
        if isinstance(model, nn.Module):
            model = torch.nn.DataParallel(model)
        elif isinstance(model, dict):
            model = {k: torch.nn.DataParallel(v) for k, v in model.items()}

    model = maybe_recursive_call(model, "to", device=device)

    return model, criterion, optimizer, scheduler, device


def get_loader(
    data_source,
    open_fn,
    dict_transform=None,
    dataset_cache_prob=-1,
    sampler=None,
    collate_fn=default_collate_fn,
    batch_size=32,
    num_workers=4,
    shuffle=False,
    drop_last=False
):
    from catalyst.data import ListDataset

    dataset = ListDataset(
        data_source,
        open_fn=open_fn,
        dict_transform=dict_transform,
        cache_prob=dataset_cache_prob
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
    "process_components", "get_loader",
    "_Model", "_Criterion", "_Optimizer", "_Scheduler"
]
