import json
import yaml
import copy
from collections import OrderedDict

from catalyst.utils.misc import create_if_need, merge_dicts

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

    # hack with argparse in config
    training_args = config.get("args", None)
    if training_args is not None:
        for key, value in training_args.items():
            arg_value = getattr(args_, key, None)
            if arg_value is None:
                arg_value = value
            setattr(args_, key, arg_value)

    if dump_config and getattr(args_, "logdir", None) is not None:
        save_config(config=config, logdir=args_.logdir)

    return args_, config


def save_config(config, logdir):
    create_if_need(logdir)
    with open("{}/config.json".format(logdir), "w") as fout:
        json.dump(config, fout, indent=2)
