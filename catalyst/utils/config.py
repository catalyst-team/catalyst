import copy
import json
import os
import platform
import shutil
import subprocess
import sys
from collections import OrderedDict
from logging import getLogger
from pathlib import Path
from typing import List, Any, Dict, Union

import safitty
import yaml
from tensorboardX import SummaryWriter

from catalyst import utils

LOG = getLogger(__name__)


def load_ordered_yaml(
    stream, Loader=yaml.Loader, object_pairs_hook=OrderedDict
):
    """
    Loads `yaml` config into OrderedDict

    Args:
        stream: opened file with yaml
        Loader: base class for yaml Loader
        object_pairs_hook: type of mapping

    Returns:
        dict: configuration
    """
    class OrderedLoader(Loader):
        pass

    def construct_mapping(loader, node):
        loader.flatten_mapping(node)
        return object_pairs_hook(loader.construct_pairs(node))

    OrderedLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, construct_mapping
    )
    return yaml.load(stream, OrderedLoader)


def _decode_dict(dictionary: Dict[str, Union[bytes, str]]) -> Dict[str, str]:
    """
    Decode bytes values in the dictionary to UTF-8
    Args:
        dictionary: a dict

    Returns:
        dict: decoded dict
    """
    result = {
        k: v.decode("UTF-8") if type(v) == bytes else v
        for k, v in dictionary.items()
    }
    return result


def get_environment_vars() -> Dict[str, Any]:
    """
    Creates a dictionary with environment variables

    Returns:
        dict: environment variables
    """
    result = {
        "python_version": sys.version,
        "conda_environment": os.environ.get("CONDA_DEFAULT_ENV", ""),
        "creation_time": utils.get_utcnow_time(),
        "sysname": platform.uname()[0],
        "nodename": platform.uname()[1],
        "release": platform.uname()[2],
        "version": platform.uname()[3],
        "architecture": platform.uname()[4],
        "user": os.environ.get("USER", ""),
        "path": os.environ.get("PWD", ""),
    }

    with open(os.devnull, "w") as devnull:
        try:
            git_branch = subprocess.check_output(
                "git rev-parse --abbrev-ref HEAD".split(),
                shell=True,
                stderr=devnull
            ).strip().decode("UTF-8")
            git_local_commit = subprocess.check_output(
                "git rev-parse HEAD".split(), shell=True, stderr=devnull
            )
            git_origin_commit = subprocess.check_output(
                f"git rev-parse origin/{git_branch}".split(),
                shell=True,
                stderr=devnull
            )

            git = dict(
                branch=git_branch,
                local_commit=git_local_commit,
                origin_commit=git_origin_commit
            )
            result["git"] = _decode_dict(git)
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass

    result = _decode_dict(result)
    return result


def list_pip_packages() -> str:
    result = ""
    with open(os.devnull, "w") as devnull:
        try:
            result = subprocess.check_output(
                "pip freeze".split(), stderr=devnull
            ).strip().decode("UTF-8")
        except FileNotFoundError:
            pass
        except subprocess.CalledProcessError as e:
            raise Exception("Failed to list packages") from e

    return result


def list_conda_packages() -> str:
    result = ""
    conda_meta_path = Path(sys.prefix) / "conda-meta"
    if conda_meta_path.exists():
        # We are currently in conda venv
        with open(os.devnull, "w") as devnull:
            try:
                result = subprocess.check_output(
                    "conda list --export".split(), stderr=devnull
                ).strip().decode("UTF-8")
            except FileNotFoundError:
                pass
            except subprocess.CalledProcessError as e:
                raise Exception(
                    "Running from conda env, but failed to list conda packages"
                ) from e
    return result


def dump_environment(
    experiment_config: Dict,
    logdir: str,
    configs_path: List[str] = None,
) -> None:
    """
    Saves config, environment variables and package list in JSON into logdir

    Args:
        experiment_config (dict): experiment config
        logdir (str): path to logdir
        configs_path: path(s) to config
    """
    configs_path = configs_path or []
    configs_path = [
        Path(path) for path in configs_path if isinstance(path, str)
    ]
    config_dir = Path(logdir) / "configs"
    config_dir.mkdir(exist_ok=True, parents=True)

    environment = get_environment_vars()

    safitty.save(experiment_config, config_dir / "_config.json")
    safitty.save(environment, config_dir / "_environment.json")

    pip_pkg = list_pip_packages()
    (config_dir / "pip-packages.txt").write_text(pip_pkg)
    conda_pkg = list_conda_packages()
    if conda_pkg:
        (config_dir / "conda-packages.txt").write_text(conda_pkg)

    for path in configs_path:
        name: str = path.name
        outpath = config_dir / name
        shutil.copyfile(path, outpath)

    config_str = json.dumps(experiment_config, indent=2, ensure_ascii=False)
    config_str = config_str.replace("\n", "\n\n")

    environment_str = json.dumps(environment, indent=2, ensure_ascii=False)
    environment_str = environment_str.replace("\n", "\n\n")

    pip_pkg = pip_pkg.replace("\n", "\n\n")
    conda_pkg = conda_pkg.replace("\n", "\n\n")
    with SummaryWriter(config_dir) as writer:
        writer.add_text("_config", config_str, 0)
        writer.add_text("_environment", environment_str, 0)
        writer.add_text("pip-packages", pip_pkg, 0)
        if conda_pkg:
            writer.add_text("conda-packages", conda_pkg, 0)


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

    args_exists_ = config.get("args")
    if args_exists_ is None:
        config["args"] = dict()

    for key, value in args._get_kwargs():
        if value is not None:
            if key in ["logdir", "baselogdir"] and value == "":
                continue
            config["args"][key] = value

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
        with open(config_path, "r") as fin:
            if config_path.endswith("json"):
                config_ = json.load(fin, object_pairs_hook=OrderedDict)
            elif config_path.endswith("yml"):
                config_ = load_ordered_yaml(fin)
            else:
                raise Exception("Unknown file format")
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


__all__ = [
    "load_ordered_yaml", "get_environment_vars", "dump_environment",
    "parse_config_args", "parse_args_uargs"
]
