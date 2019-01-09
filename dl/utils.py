import os
import copy
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
    def create_tflogger(logdir, name):
        log_dir = os.path.join(logdir, f"{name}_log")
        logger = SummaryWriter(log_dir)
        return logger

    @staticmethod
    def create_loggers(logdir, loaders):
        os.makedirs(logdir, exist_ok=True)

        loggers = []
        for key in loaders:
            logger = UtilsFactory.create_tflogger(logdir=logdir, name=key)
            loggers.append((key, logger))

        loggers = OrderedDict(loggers)

        return loggers

    @staticmethod
    def create_model(config, available_networks=None):
        # hack to prevent cycle imports
        from catalyst.contrib.models import MODELS

        model_params = config.pop("model_params", {})
        model_name = model_params.pop("model", None)
        fp16 = model_params.pop("fp16", False) and torch.cuda.is_available()

        available_networks = available_networks or {}
        available_networks = {**available_networks, **MODELS}

        model = available_networks[model_name](**model_params)

        if fp16:
            model = Fp16Wrap(model)

        return model

    @staticmethod
    def create_criterion(criterion=None, **criterion_params):
        # hack to prevent cycle imports
        from catalyst.contrib.criterion import CRITERION

        if criterion is None:
            return None
        criterion = CRITERION[criterion](**criterion_params)
        if torch.cuda.is_available():
            criterion = criterion.cuda()
        return criterion

    @staticmethod
    def create_optimizer(
        model, fp16=False, optimizer=None, **optimizer_params
    ):
        # hack to prevent cycle imports
        from catalyst.contrib.optimizers import OPTIMIZERS

        optimizer = optimizer
        if optimizer is None:
            return None

        master_params = list(
            filter(lambda p: p.requires_grad, model.parameters())
        )
        if fp16:
            assert torch.backends.cudnn.enabled, \
                "fp16 mode requires cudnn backend to be enabled."
            master_params = [
                param.detach().clone().float() for param in master_params
            ]
            for param in master_params:
                param.requires_grad = True

        optimizer = OPTIMIZERS[optimizer](master_params, **optimizer_params)
        return optimizer

    @staticmethod
    def create_scheduler(optimizer, scheduler=None, **scheduler_params):
        if optimizer is None or scheduler is None:
            return None
        scheduler = torch.optim.lr_scheduler.__dict__[scheduler](
            optimizer, **scheduler_params
        )
        return scheduler

    @staticmethod
    def create_callback(callback=None, **callback_params):
        if callback is None:
            return None

        # hack to prevent cycle imports
        from catalyst.dl.callbacks import CALLBACKS
        callback = CALLBACKS[callback](**callback_params)
        return callback

    @staticmethod
    def create_grad_clip_fn(func=None, **grad_clip_params):
        if func is None:
            return None

        func = torch.nn.utils.__dict__[func]
        grad_clip_params = copy.deepcopy(grad_clip_params)
        grad_clip_fn = lambda parameters: func(parameters, **grad_clip_params)
        return grad_clip_fn

    @staticmethod
    def prepare_model_stuff(
        model,
        criterion_params=None,
        optimizer_params=None,
        scheduler_params=None
    ):
        fp16 = isinstance(model, Fp16Wrap)

        criterion_params = criterion_params or {}
        criterion = UtilsFactory.create_criterion(**criterion_params)

        optimizer_params = optimizer_params or {}
        optimizer = UtilsFactory.create_optimizer(
            model, **optimizer_params, fp16=fp16
        )

        scheduler_params = scheduler_params or {}
        scheduler = UtilsFactory.create_scheduler(
            optimizer, **scheduler_params
        )

        return criterion, optimizer, scheduler

    @staticmethod
    def prepare_device():
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def prepare_model(model):
        device = UtilsFactory.prepare_device()

        if torch.cuda.is_available():
            cudnn.benchmark = True

        if torch.cuda.device_count() > 1 and not isinstance(model, Fp16Wrap):
            model = torch.nn.DataParallel(model).to(device)
        else:
            model = model.to(device)

        return model, device

    @staticmethod
    def prepare_stage_args(args, stage_config):
        for key, value in stage_config.get("args", {}).items():
            setattr(args, key, value)
        return args

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
    def save_checkpoint(logdir, checkpoint, is_best=False, suffix=""):
        filename = "{logdir}/checkpoint.{suffix}.pth.tar".format(
            logdir=logdir, suffix=suffix
        )
        torch.save(checkpoint, filename)
        if is_best:
            shutil.copyfile(filename, f"{logdir}/checkpoint.best.pth.tar")
        return filename

    @staticmethod
    def load_checkpoint(filepath):
        checkpoint = torch.load(
            filepath, map_location=lambda storage, loc: storage
        )
        return checkpoint
