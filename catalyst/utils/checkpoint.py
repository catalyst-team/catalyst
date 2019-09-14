from typing import Dict
import os
import shutil
from collections import OrderedDict

import torch

from .ddp import get_real_module
from .misc import maybe_recursive_call


def pack_checkpoint(
    model=None, criterion=None, optimizer=None, scheduler=None, **kwargs
):
    checkpoint = kwargs

    if isinstance(model, OrderedDict):
        raise NotImplementedError()
    else:
        model_ = get_real_module(model)
        checkpoint["model_state_dict"] = maybe_recursive_call(
            model_, "state_dict"
        )

    for dict2save, name2save in zip(
        [criterion, optimizer, scheduler],
        ["criterion", "optimizer", "scheduler"]
    ):
        if dict2save is None:
            continue
        # @TODO refactor with maybe_recursive_call
        if isinstance(dict2save, dict):
            for key, value in dict2save.items():
                if value is not None:
                    name2save_ = name2save + "_" + str(key)
                    # checkpoint[name2save_] = value
                    name2save_ = name2save_ + "_state_dict"
                    checkpoint[name2save_] = value.state_dict()
        else:
            # checkpoint[name2save] = dict2save
            name2save = name2save + "_state_dict"
            checkpoint[name2save] = dict2save.state_dict()

    return checkpoint


def save_checkpoint(
    checkpoint: Dict,
    logdir: str,
    suffix: str,
    is_best: bool = False,
    is_last: bool = False,
    special_suffix: str = ""
):
    os.makedirs(logdir, exist_ok=True)
    filename = f"{logdir}/{suffix}.pth"
    torch.save(checkpoint, filename)
    if is_best:
        shutil.copyfile(filename, f"{logdir}/best{special_suffix}.pth")
    if is_last:
        shutil.copyfile(filename, f"{logdir}/last{special_suffix}.pth")
    return filename


def unpack_checkpoint(
    checkpoint, model=None, criterion=None, optimizer=None, scheduler=None
):
    if model is not None:
        model = get_real_module(model)
        maybe_recursive_call(
            model,
            "load_state_dict",
            recursive_args=checkpoint["model_state_dict"]
        )

    for dict2load, name2load in zip(
        [criterion, optimizer, scheduler],
        ["criterion", "optimizer", "scheduler"]
    ):
        if dict2load is None:
            continue

        if isinstance(dict2load, dict):
            for key, value in dict2load.items():
                if value is not None:
                    name2load_ = f"{name2load}_{key}_state_dict"
                    value.load_state_dict(checkpoint[name2load_])
        else:
            name2load = f"{name2load}_state_dict"
            dict2load.load_state_dict(checkpoint[name2load])


def load_checkpoint(filepath):
    checkpoint = torch.load(
        filepath, map_location=lambda storage, loc: storage
    )
    return checkpoint
