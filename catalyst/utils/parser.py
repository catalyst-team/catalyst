# flake8: noqa
# @TODO: code formatting issue for 20.07 release
import copy
from pathlib import Path

from omegaconf import OmegaConf


def parse_config_args(*, config, args, unknown_args):
    """@TODO: Docs. Contribution is welcome."""
    config_args = config.get("args", None)
    if config_args is None:
        config["args"] = {}

    for key, value in args._get_kwargs():  # noqa: WPS437
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
    configs = []
    for config_path in args_copy.configs:
        configs.append(OmegaConf.load(config_path))
    cli_config = OmegaConf.from_cli()
    config = OmegaConf.merge(*configs, cli_config)

    config, args_copy = parse_config_args(
        config=config, args=args_copy, unknown_args=unknown_args
    )

    # hack with argparse in config
    config_args = config.get("args", None)
    if config_args is not None:
        for key, value in config_args.items():
            arg_value = getattr(args_copy, key, None)
            if arg_value is None or (
                key in ["logdir", "baselogdir"] and arg_value == ""
            ):
                arg_value = value
            setattr(args_copy, key, arg_value)

    return args_copy, config


__all__ = ["parse_config_args", "parse_args_uargs"]
