import os
import sys
import json
import yaml
import copy
import torch
from collections import OrderedDict
from tensorboardX import SummaryWriter
from torch.utils.data.dataloader import default_collate as default_collate_fn

from common.data.dataset import ListDataset
from common.utils import helpers
from common.utils.misc import create_if_need, stream_tee, merge_dicts
from common.utils.fp16 import network_to_half

# @TODO: for future needs; for now hyperdash is too...raw output
# try:
#     from hyperdash import Experiment
#
#     class SummaryWriter(SummaryWriter):
#         def __init__(self, log_dir=None, comment=""):
#             super().__init__(log_dir, comment)
#             log_dir = log_dir[:-1] if log_dir.endswith("/") else log_dir
#             self.exp = Experiment(log_dir.rsplit("/", 1)[-1])
#
#         def add_scalar(self, tag, scalar_value, global_step=None):
#             super().add_scalar(tag, scalar_value, global_step)
#             self.exp.metric(tag, scalar_value)
#
# except ImportError as ex:
#     print("no hyperdash support")


def parse_args_config(args, unknown_args, config):
    for arg in unknown_args:
        arg_name, value = arg.split("=")
        arg_name = arg_name[2:]
        value_content, value_type = value.rsplit(":", 1)

        if "/" in arg_name:
            arg_names = arg_name.split("/")
            if value_type == "str":
                arg_value = value_content
            else:
                arg_value = eval("%s(%s)" % (value_type, value_content))

            config_ = config
            for arg_name in arg_names[:-1]:
                if arg_name not in config_:
                    config_[arg_name] = {}

                config_ = config_[arg_name]

            config_[arg_names[-1]] = arg_value
        else:
            if value_type == "str":
                arg_value = value_content
            else:
                arg_value = eval("%s(%s)" % (value_type, value_content))
            args.__setattr__(arg_name, arg_value)
    return args, config


def load_ordered_yaml(
        stream,
        Loader=yaml.Loader, object_pairs_hook=OrderedDict):

    class OrderedLoader(Loader):
        pass

    def construct_mapping(loader, node):
        loader.flatten_mapping(node)
        return object_pairs_hook(loader.construct_pairs(node))
    OrderedLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
        construct_mapping)
    return yaml.load(stream, OrderedLoader)


def parse_args_uargs(args, unknown_args, dump_config=False):
    args_ = copy.deepcopy(args)

    # load params
    config = {}
    for config_path in args_.config.split(","):
        with open(config_path, "r") as fin:
            if config_path.endswith("json"):
                config_ = json.load(fin, object_pairs_hook=OrderedDict)
            elif config_path.endswith("yml"):
                config_ = load_ordered_yaml(fin)
            else:
                raise Exception("Unknown file format")
        config = merge_dicts(config, config_)

    args_, config = parse_args_config(args_, unknown_args, config)

    if dump_config and getattr(args_, "logdir", None) is not None:
        with open("{}/config.json".format(args_.logdir), "w") as fout:
            json.dump(config, fout, indent=2)

    # hack with argparse in config
    training_args = config.pop("args", None)
    if training_args is not None:
        for key, value in training_args.items():
            arg_value = getattr(args_, key, None)
            if arg_value is None:
                arg_value = value
            setattr(args_, key, arg_value)

    return args_, config


def create_loggers(logdir, loaders):
    create_if_need(logdir)
    logfile = open("{logdir}/log.txt".format(logdir=logdir), "w+")
    sys.stdout = stream_tee(sys.stdout, logfile)

    loggers = []
    for key in loaders:
        log_dir = os.path.join(logdir, f"{key}_log")
        logger = SummaryWriter(log_dir)
        loggers.append((key, logger))

    loggers = OrderedDict(loggers)

    return loggers


def create_model_stuff(config, available_networks):
    # create model
    model_params = config["model_params"]
    model_name = model_params.pop("model", None)
    fp16 = model_params.pop("fp16", False) and torch.cuda.is_available()
    model = available_networks[model_name](**model_params)

    if fp16:
        model = network_to_half(model)

    criterion_params = config.get("criterion_params", None) or {}
    criterion = helpers.create_criterion(**criterion_params)

    optimizer_params = config.get("optimizer_params", None) or {}
    optimizer = helpers.create_optimizer(model, **optimizer_params, fp16=fp16)

    scheduler_params = config.get("scheduler_params", None) or {}
    scheduler = helpers.create_scheduler(optimizer, **scheduler_params)

    criterion = {"main": criterion} if criterion is not None else {}
    optimizer = {"main": optimizer} if optimizer is not None else {}
    scheduler = {"main": scheduler} if scheduler is not None else {}

    criterion = OrderedDict(criterion)
    optimizer = OrderedDict(optimizer)
    scheduler = OrderedDict(scheduler)

    return model, criterion, optimizer, scheduler


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
