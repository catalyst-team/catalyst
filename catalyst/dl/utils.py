from typing import List, Tuple

import os
import shutil

from collections import OrderedDict
from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data.dataloader import default_collate as default_collate_fn

from catalyst.data.dataset import ListDataset
from catalyst.dl.fp16 import Fp16Wrap


class UtilsFactory:
    @staticmethod
    def create_loader(
        data_source,
        open_fn,
        dict_transform=None,
        dataset_cache_prob=-1,
        batch_size=32,
        workers=4,
        shuffle=False,
        sampler=None,
        collate_fn=default_collate_fn
    ):
        dataset = ListDataset(
            data_source,
            open_fn=open_fn,
            dict_transform=dict_transform,
            cache_prob=dataset_cache_prob
        )
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=workers,
            pin_memory=torch.cuda.is_available(),
            sampler=sampler,
            collate_fn=collate_fn
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
    ) -> OrderedDict[str, SummaryWriter]:
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

        if torch.cuda.device_count() > 1 and not isinstance(model, Fp16Wrap):
            model = torch.nn.DataParallel(model).to(device)
        else:
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
            model_ = model
            if isinstance(model_, nn.DataParallel):
                model_ = model_.module
            if isinstance(model_, Fp16Wrap):
                model_ = model_.network
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
            if isinstance(model, torch.nn.DataParallel):
                model = model.module
            if isinstance(model, Fp16Wrap):
                model.network.load_state_dict(checkpoint["model_state_dict"])
            else:
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
