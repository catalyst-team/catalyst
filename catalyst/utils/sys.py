from typing import Any, Dict, List, Union
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import warnings

from catalyst.contrib.utils.tools.tensorboard import SummaryWriter

from .config import save_config
from .misc import get_utcnow_time


def _decode_dict(dictionary: Dict[str, Union[bytes, str]]) -> Dict[str, str]:
    """Decode bytes values in the dictionary to UTF-8.

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
    """Creates a dictionary with environment variables.

    Returns:
        dict: environment variables
    """
    result = {
        "python_version": sys.version,
        "conda_environment": os.environ.get("CONDA_DEFAULT_ENV", ""),
        "creation_time": get_utcnow_time(),
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
            git_branch = (
                subprocess.check_output(
                    "git rev-parse --abbrev-ref HEAD".split(),
                    shell=True,
                    stderr=devnull,
                )
                .strip()
                .decode("UTF-8")
            )
            git_local_commit = subprocess.check_output(
                "git rev-parse HEAD".split(), shell=True, stderr=devnull
            )
            git_origin_commit = subprocess.check_output(
                f"git rev-parse origin/{git_branch}".split(),
                shell=True,
                stderr=devnull,
            )

            git = {
                "branch": git_branch,
                "local_commit": git_local_commit,
                "origin_commit": git_origin_commit,
            }
            result["git"] = _decode_dict(git)
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass

    result = _decode_dict(result)
    return result


def list_pip_packages() -> str:
    """
    @TODO: Docs. Contribution is welcome
    """
    result = ""
    with open(os.devnull, "w") as devnull:
        try:
            result = (
                subprocess.check_output("pip freeze".split(), stderr=devnull)
                .strip()
                .decode("UTF-8")
            )
        except Exception as e:
            warnings.warn(
                f"Failed to freeze pip packages. "
                f"Pip Output: ```{e.output}```."
                f"Continue experiment without pip packages dumping."
            )
            pass
        # except FileNotFoundError:
        #     pass
        # except subprocess.CalledProcessError as e:
        #     raise Exception("Failed to list packages") from e

    return result


def list_conda_packages() -> str:
    """
    @TODO: Docs. Contribution is welcome
    """
    result = ""
    conda_meta_path = Path(sys.prefix) / "conda-meta"
    if conda_meta_path.exists():
        # We are currently in conda virtual env
        with open(os.devnull, "w") as devnull:
            try:
                result = (
                    subprocess.check_output(
                        "conda list --export".split(), stderr=devnull
                    )
                    .strip()
                    .decode("UTF-8")
                )
            except Exception as e:
                warnings.warn(
                    f"Running from conda env, "
                    f"but failed to list conda packages. "
                    f"Conda Output: ```{e.output}```."
                    f"Continue experiment without conda packages dumping."
                )
                pass
            # except FileNotFoundError:
            #     pass
            # except subprocess.CalledProcessError as e:
            #     raise Exception(
            #         f"Running from conda env, "
            #         f"but failed to list conda packages. "
            #         f"Conda Output: {e.output}"
            #     ) from e
    return result


def dump_environment(
    experiment_config: Dict, logdir: str, configs_path: List[str] = None,
) -> None:
    """
    Saves config, environment variables and package list in JSON into logdir.

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

    save_config(experiment_config, config_dir / "_config.json")
    save_config(environment, config_dir / "_environment.json")

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


__all__ = [
    "get_environment_vars",
    "list_conda_packages",
    "list_pip_packages",
    "dump_environment",
]
