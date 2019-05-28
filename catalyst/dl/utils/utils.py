import os
import shutil
from collections import OrderedDict
from pathlib import Path
from typing import Union, Optional, List, Tuple

import safitty
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from tensorboardX import SummaryWriter
from torch.optim import Optimizer
from torch.utils.data.dataloader import default_collate as default_collate_fn

from catalyst.data.dataset import ListDataset
from catalyst.utils.plotly import plot_tensorboard_log
from catalyst.utils.model import \
    prepare_optimizable_params, assert_fp16_available


class UtilsFactory:
    prepare_optimizable_params = prepare_optimizable_params
    assert_fp16_available = assert_fp16_available

    @staticmethod
    def create_loader(
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

    @staticmethod
    def create_tflogger(logdir: str, name: str) -> SummaryWriter:
        log_dir = os.path.join(logdir, f"{name}_log")
        logger = SummaryWriter(log_dir)
        return logger

    @staticmethod
    def create_loggers(
        logdir: str, loaders: List[str]
    ) -> "OrderedDict[str, SummaryWriter]":
        os.makedirs(logdir, exist_ok=True)

        loggers = []
        for key in loaders:
            logger = UtilsFactory.create_tflogger(logdir=logdir, name=key)
            loggers.append((key, logger))

        loggers = OrderedDict(loggers)

        return loggers

    @staticmethod
    def prepare_device() -> torch.device:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def prepare_model(model: nn.Module) -> Tuple[nn.Module, torch.device]:
        device = UtilsFactory.prepare_device()

        if torch.cuda.is_available():
            cudnn.benchmark = True

        model = model.to(device)

        return model, device

    @staticmethod
    def pack_checkpoint(
        model=None, criterion=None, optimizer=None, scheduler=None, **kwargs
    ):
        checkpoint = kwargs

        if isinstance(model, OrderedDict):
            raise NotImplementedError()
        else:
            model_ = real_module_from_maybe_ddp(model)
            checkpoint["model_state_dict"] = model_.state_dict()

        for dict2save, name2save in zip(
            [criterion, optimizer, scheduler],
            ["criterion", "optimizer", "scheduler"]
        ):
            if dict2save is None:
                continue
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

    @staticmethod
    def unpack_checkpoint(
        checkpoint, model=None, criterion=None, optimizer=None, scheduler=None
    ):
        if model is not None:
            model = real_module_from_maybe_ddp(model)
            model.load_state_dict(checkpoint["model_state_dict"])

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

    @staticmethod
    def save_checkpoint(
        logdir, checkpoint, suffix="", is_best=False, is_last=False
    ):
        os.makedirs(logdir, exist_ok=True)
        filename = f"{logdir}/{suffix}.pth"
        torch.save(checkpoint, filename)
        if is_best:
            shutil.copyfile(filename, f"{logdir}/best.pth")
        if is_last:
            shutil.copyfile(filename, f"{logdir}/last.pth")
        return filename

    @staticmethod
    def load_checkpoint(filepath):
        checkpoint = torch.load(
            filepath, map_location=lambda storage, loc: storage
        )
        return checkpoint

    @staticmethod
    def plot_metrics(
        logdir: Union[str, Path],
        step: Optional[str] = "epoch",
        metrics: Optional[List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None
    ) -> None:
        """Plots your learning results.

        Args:
            logdir: the logdir that was specified during training.
            step: 'batch' or 'epoch' - what logs to show: for batches or
                for epochs
            metrics: list of metrics to plot. The loss should be specified as
                'loss', learning rate = '_base/lr' and other metrics should be
                specified as names in metrics dict
                that was specified during training
            height: the height of the whole resulting plot
            width: the width of the whole resulting plot

        """
        assert step in ["batch", "epoch"], \
            f"Step should be either 'batch' or 'epoch', got '{step}'"
        metrics = metrics or ["loss"]
        plot_tensorboard_log(logdir, step, metrics, height, width)


def get_activation_by_name(activation: str = None):
    if activation is None or activation.lower() == "none":
        activation_fn = lambda x: x
    else:
        activation_fn = torch.nn.__dict__[activation]()
    return activation_fn


def get_optimizer_momentum(optimizer: Optimizer) -> float:
    """
    Get momentum of current optimizer.

    Args:
        optimizer: PyTorch optimizer

    Returns:
        float: momentum at first param group
    """
    beta = safitty.get(optimizer.param_groups, 0, "betas", 0)
    momentum = safitty.get(optimizer.param_groups, 0, "momentum")
    return beta if beta is not None else momentum


def set_optimizer_momentum(optimizer: Optimizer, value: float, index: int = 0):
    """
    Set momentum of ``index``'th param group of optimizer to ``value``
    Args:
        optimizer: PyTorch optimizer
        value (float): new value of momentum
        index (int, optional): integer index of optimizer's param groups,
            default is 0
    """
    betas = safitty.get(optimizer.param_groups, index, "betas")
    momentum = safitty.get(optimizer.param_groups, index, "momentum")
    if betas is not None:
        _, beta = betas
        safitty.set(
            optimizer.param_groups, index, "betas", value=(value, beta)
        )
    elif momentum is not None:
        safitty.set(optimizer.param_groups, index, "momentum", value=value)


def is_wrapped_with_ddp(model: nn.Module) -> bool:
    """
    Checks whether model is wrapped with DataParallel/DistributedDataParallel.
    :param model:
    :return:
    """
    parallel_wrappers = torch.nn.DataParallel, \
        torch.nn.parallel.DistributedDataParallel

    # Check whether Apex is installed and if it is,
    # add Apex's DistributedDataParallel to list of checked types
    try:
        from apex.parallel import DistributedDataParallel as apex_DDP
        parallel_wrappers = parallel_wrappers + (apex_DDP, )
    except ImportError:
        pass

    return isinstance(model, parallel_wrappers)


def real_module_from_maybe_ddp(model: nn.Module) -> nn.Module:
    """
    Return a real model from a torch.nn.DataParallel,
    torch.nn.parallel.DistributedDataParallel, or
    apex.parallel.DistributedDataParallel.

    :param model: A model, or DataParallel wrapper.
    :return: A model
    """
    if is_wrapped_with_ddp(model):
        model = model.module
    return model
