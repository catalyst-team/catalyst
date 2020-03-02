import copy
from pathlib import Path

from catalyst import utils


def parse_config_args(*, config, args, unknown_args):
    for arg in unknown_args:
        arg_name, value = arg.split("=")
        arg_name = arg_name.lstrip("-").strip("/")

        value_content, value_type = value.rsplit(":", 1)

        if "/" in arg_name:
            arg_names = arg_name.split("/")
            if value_type == "str":
                arg_value = value_content

                if arg_value.lower() == "none":
                    arg_value = None
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

    args_exists_ = config.get("args", None)
    if args_exists_ is None:
        config["args"] = dict()

    for key, value in args._get_kwargs():
        if value is not None:
            if key in ["logdir", "baselogdir"] and value == "":
                continue
            config["args"][key] = value

    autoresume = config["args"].get("autoresume", None)
    logdir = config["args"].get("logdir", None)
    resume = config["args"].get("resume", None)
    if autoresume is not None and logdir is not None and resume is None:
        logdir = Path(logdir)
        checkpoint_filename = logdir / "checkpoints" / f"{autoresume}_full.pth"
        if checkpoint_filename.is_file():
            config["args"]["resume"] = str(checkpoint_filename)
    return config, args


def parse_args_uargs(args, unknown_args):
    """
    Function for parsing configuration files

    Args:
        args: recognized arguments
        unknown_args: unrecognized arguments

    Returns:
        tuple: updated arguments, dict with config
    """
    args_ = copy.deepcopy(args)

    # load params
    config = {}
    for config_path in args_.configs:
        config_ = utils.load_config(config_path, ordered=True)
        config = utils.merge_dicts(config, config_)

    config, args_ = parse_config_args(
        config=config, args=args_, unknown_args=unknown_args
    )

    # hack with argparse in config
    config_args = config.get("args", None)
    if config_args is not None:
        for key, value in config_args.items():
            arg_value = getattr(args_, key, None)
            if arg_value is None or \
                    (key in ["logdir", "baselogdir"] and arg_value == ""):
                arg_value = value
            setattr(args_, key, arg_value)

    return args_, config


__all__ = ["parse_config_args", "parse_args_uargs"]
