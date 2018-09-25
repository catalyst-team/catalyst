import os
import sys
import shutil
from collections import OrderedDict
from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data.dataloader import default_collate as default_collate_fn

from prometheus.optimizers.optimizers import OPTIMIZERS
from prometheus.losses.losses import LOSSES
from prometheus.data.dataset import ListDataset
from prometheus.utils.fp16 import Fp16Wrap, network_to_half
from prometheus.utils.misc import create_if_need, stream_tee


class UtilsFactory:

    @staticmethod
    def create_loader(
            data_source, open_fn,
            dict_transform=None, dataset_cache_prob=-1,
            batch_size=32, workers=4,
            shuffle=False, sampler=None, collate_fn=default_collate_fn):
        dataset = ListDataset(
            data_source, open_fn=open_fn,
            dict_transform=dict_transform,
            cache_prob=dataset_cache_prob)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=workers,
            pin_memory=torch.cuda.is_available(),
            sampler=sampler,
            collate_fn=collate_fn)
        return loader

    @staticmethod
    def create_loggers(logdir, loaders):
        create_if_need(logdir)
        logfile = open("{logdir}/log.txt".format(logdir=logdir), "a")
        sys.stdout = stream_tee(sys.stdout, logfile)

        loggers = []
        for key in loaders:
            log_dir = os.path.join(logdir, f"{key}_log")
            logger = SummaryWriter(log_dir)
            loggers.append((key, logger))

        loggers = OrderedDict(loggers)

        return loggers

    @staticmethod
    def create_model(config, available_networks):
        model_params = config["model_params"]
        model_name = model_params.pop("model", None)
        fp16 = model_params.pop("fp16", False) and torch.cuda.is_available()
        model = available_networks[model_name](**model_params)

        if fp16:
            model = network_to_half(model)

        return model

    @staticmethod
    def create_criterion(criterion=None, **criterion_params):
        if criterion is None:
            return None
        criterion = LOSSES[criterion](**criterion_params)
        if torch.cuda.is_available():
            criterion = criterion.cuda()
        return criterion

    @staticmethod
    def create_optimizer(
            model, fp16=False, optimizer=None, **optimizer_params):
        optimizer = optimizer
        if optimizer is None:
            return None

        master_params = list(
            filter(lambda p: p.requires_grad, model.parameters()))
        if fp16:
            assert torch.backends.cudnn.enabled, \
                "fp16 mode requires cudnn backend to be enabled."
            master_params = [
                param.detach().clone().float()
                for param in master_params]
            for param in master_params:
                param.requires_grad = True

        optimizer = OPTIMIZERS[optimizer](
            master_params, **optimizer_params)
        return optimizer

    @staticmethod
    def create_scheduler(optimizer, scheduler=None, **scheduler_params):
        if optimizer is None or scheduler is None:
            return None
        scheduler = torch.optim.lr_scheduler.__dict__[scheduler](
            optimizer, **scheduler_params)
        return scheduler

    @staticmethod
    def create_model_stuff(model, config):
        fp16 = isinstance(model, Fp16Wrap)

        criterion_params = config.get("criterion_params", None) or {}
        criterion = UtilsFactory.create_criterion(**criterion_params)

        optimizer_params = config.get("optimizer_params", None) or {}
        optimizer = UtilsFactory.create_optimizer(
            model, **optimizer_params, fp16=fp16)

        scheduler_params = config.get("scheduler_params", None) or {}
        scheduler = UtilsFactory.create_scheduler(
            optimizer, **scheduler_params)

        criterion = {"main": criterion} if criterion is not None else {}
        optimizer = {"main": optimizer} if optimizer is not None else {}
        scheduler = {"main": scheduler} if scheduler is not None else {}

        criterion = OrderedDict(criterion)
        optimizer = OrderedDict(optimizer)
        scheduler = OrderedDict(scheduler)

        return criterion, optimizer, scheduler

    @staticmethod
    def prepare_model(model):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    def prepare_stage_stuff(model, stage_config):
        return UtilsFactory.create_model_stuff(
            model=model, config=stage_config)

    @staticmethod
    def get_val_from_metric(metric_value):
        if isinstance(metric_value, (int, float)):
            pass
        elif torch.is_tensor(metric_value):
            metric_value = metric_value.item()
        else:
            metric_value = metric_value.value()
            if isinstance(metric_value, (tuple, list)):
                metric_value = metric_value[0]
            if torch.is_tensor(metric_value):
                metric_value = metric_value.item()
        return metric_value

    @staticmethod
    def pack_checkpoint(
            model=None,
            criterion=None, optimizer=None, scheduler=None,
            **kwargs):
        checkpoint = kwargs

        if isinstance(model, OrderedDict):
            raise NotImplementedError()
            for key, value in model.items():
                name2save = key + "_model_state_dict"
                checkpoint[name2save] = value.state_dict()
        else:
            model_ = model
            if isinstance(model_, nn.DataParallel):
                model_ = model_.module
            if isinstance(model_, Fp16Wrap):
                model_ = model_.network
            checkpoint["model_state_dict"] = model_.state_dict()

        for dict2save, name2save in zip(
                [criterion, optimizer, scheduler],
                ["criterion", "optimizer", "scheduler"]):
            for key, value in dict2save.items():
                if value is not None:
                    name2save_ = name2save + "_" + str(key)
                    checkpoint[name2save_] = value
                    name2save_ = name2save_ + "_state_dict"
                    checkpoint[name2save_] = value.state_dict()

        return checkpoint

    @staticmethod
    def save_checkpoint(logdir, checkpoint, is_best=False, suffix=""):
        filename = "{logdir}/checkpoint.{suffix}.pth.tar".format(
            logdir=logdir, suffix=suffix)
        torch.save(checkpoint, filename)
        if is_best:
            shutil.copyfile(
                filename, "{}/checkpoint.best.pth.tar".format(logdir))
        return filename

    @staticmethod
    def load_checkpoint(filepath):
        checkpoint = torch.load(
            filepath,
            map_location=lambda storage, loc: storage)
        return checkpoint

    @staticmethod
    def unpack_checkpoint(
            checkpoint,
            model=None, criterion=None, optimizer=None, scheduler=None):
        if model is not None:
            if isinstance(model, torch.nn.DataParallel):
                model = model.module
            if isinstance(model, Fp16Wrap):
                model.network.load_state_dict(
                    checkpoint["model_state_dict"])
            else:
                model.load_state_dict(
                    checkpoint["model_state_dict"])

        if criterion is not None:
            for key in criterion:
                criterion[key].load_state_dict(
                    checkpoint["criterion_" + str(key) + "_state_dict"])

        if optimizer is not None:
            for key in optimizer:
                optimizer[key].load_state_dict(
                    checkpoint["optimizer_" + str(key) + "_state_dict"])

        if scheduler is not None:
            for key in scheduler:
                scheduler[key] = checkpoint["scheduler_" + str(key)]
