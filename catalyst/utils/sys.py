from typing import Any, Dict, List, Union
import copy
from importlib.util import module_from_spec, spec_from_file_location
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
from subprocess import CalledProcessError
import sys
import warnings

from tensorboardX import SummaryWriter

from catalyst.registry import REGISTRY
from catalyst.settings import SETTINGS
from catalyst.utils.config import save_config
from catalyst.utils.misc import get_utcnow_time

if SETTINGS.hydra_required:
    from omegaconf import DictConfig, OmegaConf


def import_module(expdir: Union[str, Path]):
    """
    Imports python module by path.

    Args:
        expdir: path to python module.

    Returns:
        Imported module.
    """
    if not isinstance(expdir, Path):
        expdir = Path(expdir)
    sys.path.insert(0, str(expdir.absolute()))
    sys.path.insert(0, os.path.dirname(str(expdir.absolute())))
    module_spec = spec_from_file_location(
        expdir.name,
        str(expdir.absolute() / "__init__.py"),
        submodule_search_locations=[expdir.absolute()],
    )
    dir_module = module_from_spec(module_spec)
    module_spec.loader.exec_module(dir_module)
    sys.modules[expdir.name] = dir_module
    return dir_module


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
    if not isinstance(expdir, Path):
        expdir = Path(expdir)
    dir_module = import_module(expdir)  # noqa: F841
    # runner_fn = getattr(m, "Runner", None)

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


def _tricky_dir_copy(dir_from: str, dir_to: str) -> None:
    os.makedirs(dir_to, exist_ok=True)
    shutil.rmtree(dir_to)
    shutil.copytree(dir_from, dir_to)


def dump_code(logdir: Union[str, Path], expdir: Union[str, Path] = None):
    """
    Dumps Catalyst code for reproducibility.

    Args:
        logdir: logging dir path
        expdir: experiment dir path
    """
    new_src_dir = "code"

    # @TODO: hardcoded
    old_pro_dir = os.path.dirname(os.path.abspath(__file__)) + "/../"
    new_pro_dir = os.path.join(logdir, new_src_dir, "catalyst")
    _tricky_dir_copy(old_pro_dir, new_pro_dir)

    if expdir is not None:
        expdir = expdir[:-1] if expdir.endswith("/") else expdir
        old_expdir = os.path.abspath(expdir)
        new_expdir = os.path.basename(old_expdir)
        new_expdir = os.path.join(logdir, new_src_dir, new_expdir)
        _tricky_dir_copy(old_expdir, new_expdir)


def _decode_dict(dictionary: Dict[str, Union[bytes, str]]) -> Dict[str, str]:
    """Decode bytes values in the dictionary to UTF-8."""
    result = {k: v.decode("UTF-8") if type(v) == bytes else v for k, v in dictionary.items()}
    return result


def _get_environment_vars() -> Dict[str, Any]:
    """Creates a dictionary with environment variables."""
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
                    "git rev-parse --abbrev-ref HEAD".split(), shell=True, stderr=devnull,
                )
                .strip()
                .decode("UTF-8")
            )
            git_local_commit = subprocess.check_output(
                "git rev-parse HEAD".split(), shell=True, stderr=devnull
            )
            git_origin_commit = subprocess.check_output(
                f"git rev-parse origin/{git_branch}".split(), shell=True, stderr=devnull,
            )

            git = {
                "branch": git_branch,
                "local_commit": git_local_commit,
                "origin_commit": git_origin_commit,
            }
            result["git"] = _decode_dict(git)
        except (CalledProcessError, FileNotFoundError):
            pass

    result = _decode_dict(result)
    return result


def _list_pip_packages() -> str:
    """Lists pip installed packages."""
    result = ""
    with open(os.devnull, "w") as devnull:
        try:
            result = (
                subprocess.check_output("pip freeze".split(), stderr=devnull)
                .strip()
                .decode("UTF-8")
            )
        except Exception:
            warnings.warn(
                "Failed to freeze pip packages. "
                # f"Pip Output: ```{e.output}```."
                "Continue run without pip packages dumping."
            )
            pass
        # except FileNotFoundError:
        #     pass
        # except subprocess.CalledProcessError as e:
        #     raise Exception("Failed to list packages") from e

    return result


def _list_conda_packages() -> str:
    """Lists conda installed packages."""
    result = ""
    conda_meta_path = Path(sys.prefix) / "conda-meta"
    if conda_meta_path.exists():
        # We are currently in conda virtual env
        with open(os.devnull, "w") as devnull:
            try:
                result = (
                    subprocess.check_output("conda list --export".split(), stderr=devnull)
                    .strip()
                    .decode("UTF-8")
                )
            except Exception:
                warnings.warn(
                    "Running from conda env, "
                    "but failed to list conda packages. "
                    # f"Conda Output: ```{e.output}```."
                    "Continue run without conda packages dumping."
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


def dump_environment(logdir: str, config: Any = None, configs_path: List[str] = None) -> None:
    """
    Saves config, environment variables and package list in JSON into logdir.

    Args:
        logdir: path to logdir
        config: experiment config
        configs_path: path(s) to config
    """
    configs_path = configs_path or []
    configs_path = [Path(path) for path in configs_path if isinstance(path, str)]
    config_dir = Path(logdir) / "configs"
    config_dir.mkdir(exist_ok=True, parents=True)

    if SETTINGS.hydra_required and isinstance(config, DictConfig):
        with open(config_dir / "config.yaml", "w") as f:
            f.write(OmegaConf.to_yaml(config, resolve=True))
        config = OmegaConf.to_container(config, resolve=True)

    environment = _get_environment_vars()
    save_config(environment, config_dir / "_environment.json")
    if config is not None:
        save_config(config, config_dir / "_config.json")

    pip_pkg = _list_pip_packages()
    (config_dir / "pip-packages.txt").write_text(pip_pkg)
    conda_pkg = _list_conda_packages()
    if conda_pkg:
        (config_dir / "conda-packages.txt").write_text(conda_pkg)

    for path in configs_path:
        name: str = path.name
        outpath = config_dir / name
        shutil.copyfile(path, outpath)

    pip_pkg = pip_pkg.replace("\n", "\n\n")
    conda_pkg = conda_pkg.replace("\n", "\n\n")
    with SummaryWriter(config_dir) as writer:
        if config is not None:
            config_str = json.dumps(config, indent=2, ensure_ascii=False)
            config_str = config_str.replace("\n", "\n\n")
            writer.add_text("_config", config_str, 0)

        environment_str = json.dumps(environment, indent=2, ensure_ascii=False)
        environment_str = environment_str.replace("\n", "\n\n")
        writer.add_text("_environment", environment_str, 0)

        writer.add_text("pip-packages", pip_pkg, 0)
        if conda_pkg:
            writer.add_text("conda-packages", conda_pkg, 0)


__all__ = [
    "dump_environment",
    "import_module",
    "dump_code",
    "get_config_runner",
]
