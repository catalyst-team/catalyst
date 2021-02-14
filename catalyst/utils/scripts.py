from typing import Dict, Union
import copy
from importlib.util import module_from_spec, spec_from_file_location
import os
import pathlib
import shutil
import sys

from catalyst.registry import REGISTRY


def import_module(expdir: Union[str, pathlib.Path]):
    """
    Imports python module by path.

    Args:
        expdir: path to python module.

    Returns:
        Imported module.
    """
    if not isinstance(expdir, pathlib.Path):
        expdir = pathlib.Path(expdir)
    sys.path.insert(0, str(expdir.absolute()))
    sys.path.insert(0, os.path.dirname(str(expdir.absolute())))
    s = spec_from_file_location(
        expdir.name,
        str(expdir.absolute() / "__init__.py"),
        submodule_search_locations=[expdir.absolute()],
    )
    m = module_from_spec(s)
    s.loader.exec_module(m)
    sys.modules[expdir.name] = m
    return m


def get_config_runner(expdir: pathlib.Path, config: Dict):
    """
    Imports and creates ConfigRunner instance.

    Args:
        expdir: experiment directory path
        config: dictionary with experiment Config

    Returns:
        ConfigRunner instance
    """
    config_copy = copy.deepcopy(config)
    if not isinstance(expdir, pathlib.Path):
        expdir = pathlib.Path(expdir)
    m = import_module(expdir)
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


def dump_code(expdir: Union[str, pathlib.Path], logdir: Union[str, pathlib.Path]) -> None:
    """
    Dumps Catalyst code for reproducibility.

    Args:
        expdir (Union[str, pathlib.Path]): experiment dir path
        logdir (Union[str, pathlib.Path]): logging dir path
    """
    expdir = expdir[:-1] if expdir.endswith("/") else expdir
    new_src_dir = "code"

    # @TODO: hardcoded
    old_pro_dir = os.path.dirname(os.path.abspath(__file__)) + "/../"
    new_pro_dir = os.path.join(logdir, new_src_dir, "catalyst")
    _tricky_dir_copy(old_pro_dir, new_pro_dir)

    old_expdir = os.path.abspath(expdir)
    new_expdir = os.path.basename(old_expdir)
    new_expdir = os.path.join(logdir, new_src_dir, new_expdir)
    _tricky_dir_copy(old_expdir, new_expdir)


# def _dump_pyfiles(src: pathlib.Path, dst: pathlib.Path) -> None:
#     """Dumps python code (``*.py`` and ``*.ipynb``) files."""
#     py_files = list(src.glob("*.py"))
#     ipynb_files = list(src.glob("*.ipynb"))
#
#     py_files += ipynb_files
#     py_files = list(set(py_files))
#     for py_file in py_files:
#         shutil.copy2(f"{str(py_file.absolute())}", f"{dst}/{py_file.name}")


# def dump_experiment_code(src: pathlib.Path, dst: pathlib.Path) -> None:
#     """
#     Dumps your experiment code for Config API use cases.
#
#     Args:
#         src: source code path
#         dst: destination code path
#     """
#     utcnow = get_utcnow_time()
#     dst = dst.joinpath("code")
#     dst = dst.joinpath(f"code-{utcnow}") if dst.exists() else dst
#     os.makedirs(dst, exist_ok=True)
#     _dump_pyfiles(src, dst)


# def distributed_cmd_run(worker_fn: Callable, distributed: bool = True, *args, **kwargs) -> None:
#     """
#     Distributed run
#
#     Args:
#         worker_fn: worker fn to run in distributed mode
#         distributed: distributed flag
#         args: additional parameters for worker_fn
#         kwargs: additional key-value parameters for worker_fn
#     """
#     distributed_params = get_distributed_params()
#     local_rank = distributed_params["local_rank"]
#     world_size = distributed_params["world_size"]
#
#     if distributed and torch.distributed.is_initialized():
#         warnings.warn(
#             "Looks like you are trying to call distributed setup twice, "
#             "switching to normal run for correct distributed training."
#         )
#
#     if not distributed or torch.distributed.is_initialized() or world_size <= 1:
#         worker_fn(*args, **kwargs)
#     elif local_rank is not None:
#         torch.cuda.set_device(int(local_rank))
#
#         torch.distributed.init_process_group(backend="nccl", init_method="env://")
#         worker_fn(*args, **kwargs)
#     else:
#         workers = []
#         try:
#             for local_rank in range(torch.cuda.device_count()):
#                 rank = distributed_params["start_rank"] + local_rank
#                 env = get_distributed_env(local_rank, rank, world_size)
#                 cmd = [sys.executable] + sys.argv.copy()
#                 workers.append(subprocess.Popen(cmd, env=env))
#             for worker in workers:
#                 worker.wait()
#         finally:
#             for worker in workers:
#                 worker.kill()


__all__ = [
    "import_module",
    "dump_code",
    "get_config_runner",
]
