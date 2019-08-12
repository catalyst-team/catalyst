from typing import Tuple, Dict, List, Union
import os
import copy

import numpy as np

import torch
from torch import nn, optim
import torch.backends.cudnn as cudnn
from torch.utils.data.dataloader import default_collate as default_collate_fn

from catalyst.dl import utils

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

    if torch.cuda.is_available():
        benchmark = os.environ.get("CUDNN_BENCHMARK", "True") == "True"
        cudnn.benchmark = benchmark

    model = model.to(device)

    if utils.is_wrapped_with_ddp(model):
        pass
    elif len(distributed_params) > 0:
        utils.assert_fp16_available()
        from apex import amp

        distributed_rank = distributed_params.pop("rank", -1)

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
        elif torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
    elif torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    model = model.to(device)

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


def get_model_params(
    model: _Model,
    weight_decay: float = 0.0,
    no_bias_weight_decay: bool = True
) -> List[Union[torch.nn.Parameter, dict]]:
    """
    Gains model parameters for ``torch.optim.Optimizer``

    Args:
        model (torch.nn.Module): Model to process
        weight_decay (float): Optional weight decay
        no_bias_weight_decay (bool): If true, removes weight_decay
            for all ``bias`` parameters in the model

    Returns:
        iterable: parameters for an optimizer

    Examples:
        >>> model = ResnetUnet()
        >>> params = get_model_params(model, weight_decay=0.00001)
        >>> optimizer = torch.optim.Adam(params, lr=0.0003)
    """
    params = list(model.named_parameters())

    if not no_bias_weight_decay or np.isclose(weight_decay, 0.0):
        return [param for (name, param) in params]

    # no bias decay from https://arxiv.org/abs/1812.01187
    biases = [param for (name, param) in params if name.endswith("bias")]
    main_params = [
        param for (name, param) in params if not name.endswith("bias")
    ]

    result = [
        {"params": main_params, "weight_decay": weight_decay},
        {"params": biases, "weight_decay": 0.0},
    ]

    return result


__all__ = [
    "process_components", "get_loader", "get_model_params",
    "_Model", "_Criterion", "_Optimizer", "_Scheduler"
]
