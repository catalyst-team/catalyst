from typing import Callable, Dict, Union
import os
from pathlib import Path
import shutil

import torch
from torch import nn

from catalyst.utils.distributed import get_nn_from_ddp_module
from catalyst.utils.misc import maybe_recursive_call


def pack_checkpoint(
    model: nn.Module = None, criterion: nn.Module = None, optimizer=None, scheduler=None, **kwargs,
):
    """
    Packs ``model``, ``criterion``, ``optimizer``, ``scheduler``
    and some extra info ``**kwargs`` to torch-based checkpoint.

    Args:
        model: torch model
        criterion: torch criterion
        optimizer: torch optimizer
        scheduler: torch scheduler
        **kwargs: some extra info to pack

    Returns:
        torch-based checkpoint with ``model_state_dict``,
        ``criterion_state_dict``, ``optimizer_state_dict``,
        ``scheduler_state_dict`` keys.
    """
    checkpoint = kwargs

    if isinstance(model, dict):
        for key, value in model.items():
            model_module = get_nn_from_ddp_module(value)
            checkpoint[f"model_{key}_state_dict"] = maybe_recursive_call(
                model_module, "state_dict"
            )
    else:
        model_module = get_nn_from_ddp_module(model)
        checkpoint["model_state_dict"] = maybe_recursive_call(model_module, "state_dict")

    for dict2save, name2save in zip(
        [criterion, optimizer, scheduler], ["criterion", "optimizer", "scheduler"],
    ):
        if dict2save is None:
            continue
        # @TODO refactor with maybe_recursive_call (?)
        if isinstance(dict2save, dict):
            for key, value in dict2save.items():
                if value is not None:
                    state_dict2save = name2save + "_" + str(key)
                    # checkpoint[name2save_] = value
                    state_dict2save = state_dict2save + "_state_dict"
                    checkpoint[state_dict2save] = value.state_dict()
        else:
            # checkpoint[name2save] = dict2save
            name2save = name2save + "_state_dict"
            checkpoint[name2save] = dict2save.state_dict()

    return checkpoint


def unpack_checkpoint(
    checkpoint, model=None, criterion=None, optimizer=None, scheduler=None
) -> None:
    """Load checkpoint from file and unpack the content to a model
    (if not None), criterion (if not None), optimizer (if not None),
    scheduler (if not None).

    Args:
        checkpoint: checkpoint to load
        model: model where should be updated state
        criterion: criterion where should be updated state
        optimizer: optimizer where should be updated state
        scheduler: scheduler where should be updated state
    """
    if model is not None:
        model = get_nn_from_ddp_module(model)
        maybe_recursive_call(
            model, "load_state_dict", recursive_args=checkpoint["model_state_dict"],
        )

    for dict2load, name2load in zip(
        [criterion, optimizer, scheduler], ["criterion", "optimizer", "scheduler"],
    ):
        if dict2load is None:
            continue

        if isinstance(dict2load, dict):
            for key, value in dict2load.items():
                if value is not None:
                    state_dict2load = f"{name2load}_{key}_state_dict"
                    value.load_state_dict(checkpoint[state_dict2load])
        else:
            name2load = f"{name2load}_state_dict"
            dict2load.load_state_dict(checkpoint[name2load])


def save_checkpoint(
    checkpoint: Dict,
    logdir: Union[Path, str],
    suffix: str,
    is_best: bool = False,
    is_last: bool = False,
    special_suffix: str = "",
    saver_fn: Callable = torch.save,
) -> Union[Path, str]:
    """Saving checkpoint to a file.

    Args:
        checkpoint: data to save.
        logdir: directory where checkpoint
            should be stored.
        suffix: checkpoint file name.
        is_best: if ``True`` then also
            will be generated best checkpoint file.
        is_last: if ``True`` then also
            will be generated last checkpoint file.
        special_suffix: suffix to use for
            saving best/last checkpoints.
        saver_fn: function to use for saving
            data to file, default is ``torch.save``

    Returns:
        path to saved checkpoint
    """
    os.makedirs(logdir, exist_ok=True)
    filename = f"{logdir}/{suffix}.pth"
    saver_fn(checkpoint, filename)
    if is_best:
        shutil.copyfile(filename, f"{logdir}/best{special_suffix}.pth")
    if is_last:
        shutil.copyfile(filename, f"{logdir}/last{special_suffix}.pth")
    return filename


def load_checkpoint(filepath: str):
    """Load checkpoint from path.

    Args:
        filepath: checkpoint file to load

    Returns:
        checkpoint content
    """
    checkpoint = torch.load(filepath, map_location=lambda storage, loc: storage)
    return checkpoint


__all__ = [
    "pack_checkpoint",
    "unpack_checkpoint",
    "save_checkpoint",
    "load_checkpoint",
]
