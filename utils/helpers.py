import shutil
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

from common.optimizers.optimizers import OPTIMIZERS
from common.losses.losses import LOSSES
from common.utils.fp16 import Fp16Wrap


def create_criterion(**criterion_params):
    criterion_name = criterion_params.pop("criterion", None)
    if criterion_name is None:
        return None
    criterion = LOSSES[criterion_name](**criterion_params)
    if torch.cuda.is_available():
        criterion = criterion.cuda()
    return criterion


def create_optimizer(model, fp16=False, **optimizer_params):
    optimizer_name = optimizer_params.pop("optimizer", None)
    if optimizer_name is None:
        return None

    master_params = list(filter(lambda p: p.requires_grad, model.parameters()))
    if fp16:
        assert torch.backends.cudnn.enabled, \
            "fp16 mode requires cudnn backend to be enabled."
        master_params = [
            param.detach().clone().float()
            for param in master_params]
        for param in master_params:
            param.requires_grad = True

    optimizer = OPTIMIZERS[optimizer_name](master_params, **optimizer_params)
    return optimizer


def create_scheduler(optimizer, **scheduler_params):
    if optimizer is None:
        return None
    scheduler_name = scheduler_params.pop("scheduler", None)
    if scheduler_name is None:
        return None
    scheduler = torch.optim.lr_scheduler.__dict__[scheduler_name](
        optimizer, **scheduler_params)
    return scheduler


def prepare_model(model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        cudnn.benchmark = True

    if torch.cuda.device_count() > 1 and not isinstance(model, Fp16Wrap):
        model = torch.nn.DataParallel(model).to(device)
    else:
        model = model.to(device)

    return model, device


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


def save_checkpoint(logdir, checkpoint, is_best, suffix=""):
    filename = "{logdir}/checkpoint.{suffix}.pth.tar".format(
        logdir=logdir, suffix=suffix)
    torch.save(checkpoint, filename)
    if is_best:
        shutil.copyfile(filename, "{}/checkpoint.best.pth.tar".format(logdir))
    return filename


def load_checkpoint(filepath):
    checkpoint = torch.load(
        filepath,
        map_location=lambda storage, loc: storage)
    return checkpoint


def prepare_checkpoint(
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
