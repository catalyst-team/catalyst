from typing import Dict
import copy
from pathlib import Path

from catalyst.registry import REGISTRY
from catalyst.utils.config import load_config
from catalyst.utils.misc import merge_dicts
from catalyst.utils.sys import import_module


def parse_config_args(*, config, args, unknown_args):
    """Parse config and cli args.

    Args:
        config: dict-based experiment config
        args: cli args
        unknown_args: cli unknown args

    Returns:
        config, args: final experiment config and cli args
    """
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

            config_copy = config
            for arg_name in arg_names[:-1]:
                if arg_name not in config_copy:
                    config_copy[arg_name] = {}

                config_copy = config_copy[arg_name]

            config_copy[arg_names[-1]] = arg_value
        else:
            if value_type == "str":
                arg_value = value_content
            else:
                arg_value = eval("%s(%s)" % (value_type, value_content))
            args.__setattr__(arg_name, arg_value)

    config_args = config.get("args", None)
    if config_args is None:
        config["args"] = {}

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
    """Function for parsing configuration files.

    Args:
        args: recognized arguments
        unknown_args: unrecognized arguments

    Returns:
        tuple: updated arguments, dict with config
    """
    args_copy = copy.deepcopy(args)

    # load params
    config = {}
    for config_path in args_copy.configs:
        config_part = load_config(config_path, ordered=True)
        config = merge_dicts(config, config_part)

    config, args_copy = parse_config_args(config=config, args=args_copy, unknown_args=unknown_args)

    # hack with argparse in config
    config_args = config.get("args", None)
    if config_args is not None:
        for key, value in config_args.items():
            arg_value = getattr(args_copy, key, None)
            if arg_value is None or (key in ["logdir", "baselogdir"] and arg_value == ""):
                arg_value = value
            setattr(args_copy, key, arg_value)

    return args_copy, config


def get_config_runner(expdir: Path, config: Dict):
    """
    Imports and creates ConfigRunner instance.

    Args:
        expdir: experiment directory path
        config: dictionary with experiment Config

    Returns:
        ConfigRunner instance
    """
    config_copy = copy.deepcopy(config)

    if expdir is not None:
        dir_module = import_module(expdir)  # noqa: F841
        # runner_fn = getattr(dir_module, "Runner", None)

    runner_params = config_copy.get("runner", {})
    runner_from_config = runner_params.pop("_target_", None)
    assert runner_from_config is not None, "You should specify the ConfigRunner."
    runner_fn = REGISTRY.get(runner_from_config)
    # assert any(
    #     x is None for x in (runner_fn, runner_from_config)
    # ), "Runner is set both in code and config."
    # if runner_fn is None and runner_from_config is not None:
    #     runner_fn = REGISTRY.get(runner_from_config)

    runner = runner_fn(config=config_copy, **runner_params)

    return runner


__all__ = ["parse_config_args", "parse_args_uargs", "get_config_runner"]
