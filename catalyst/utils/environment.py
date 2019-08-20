import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Union

import safitty
from tensorboardX import SummaryWriter


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
        "creation_time": time.strftime("%y-%m-%d-%H-%M-%S"),
        "sysname": os.uname()[0],
        "nodename": os.uname()[1],
        "release": os.uname()[2],
        "version": os.uname()[3],
        "architecture": os.uname()[4],
        "user": os.environ["USER"],
        "path": os.environ["PWD"]
    }

    with open(os.devnull, "w") as devnull:
        try:
            git_branch = subprocess.check_output(
                "git rev-parse --abbrev-ref HEAD".split(), stderr=devnull
            ).strip().decode("UTF-8")
            git_local_commit = subprocess.check_output(
                "git rev-parse HEAD".split(), stderr=devnull
            )
            git_origin_commit = subprocess.check_output(
                f"git rev-parse origin/{git_branch}".split(), stderr=devnull
            )

            git = dict(
                branch=git_branch,
                local_commit=git_local_commit,
                origin_commit=git_origin_commit
            )
            result["git"] = _decode_dict(git)
        except subprocess.CalledProcessError:
            pass

    result = _decode_dict(result)
    return result


def dump_environment_vars(
    environment: Dict[str, Any],
    logdir: str,
) -> None:
    """
    Saves environment variables in JSON into logdir

    Args:
        environment (dict): dictionary with environment variables to dump
        logdir (str): path to logdir
    """
    config_dir = Path(logdir) / "environment"
    config_dir.mkdir(exist_ok=True, parents=True)

    safitty.save(environment, config_dir / "_environment.json")

    environment_str = json.dumps(environment, indent=2)
    environment_str = environment_str.replace("\n", "\n\n")
    with SummaryWriter(config_dir) as writer:
        writer.add_text("environment", environment_str, 0)


__all__ = ["get_environment_vars", "dump_environment_vars"]
