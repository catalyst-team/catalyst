import os
import pathlib
from catalyst.utils.misc import get_utcnow_time
from catalyst.utils.scripts import import_module, dump_python_files


def import_experiment_and_runner(expdir: pathlib.Path):
    if not isinstance(expdir, pathlib.Path):
        expdir = pathlib.Path(expdir)
    m = import_module(expdir)
    Experiment, Runner = m.Experiment, m.Runner
    return Experiment, Runner


def dump_base_experiment_code(src: pathlib.Path, dst: pathlib.Path):
    utcnow = get_utcnow_time()
    dst_ = dst.joinpath("code")
    dst = dst.joinpath(f"code-{utcnow}") if dst_.exists() else dst_
    os.makedirs(dst, exist_ok=True)
    dump_python_files(src, dst)


__all__ = ["import_experiment_and_runner", "dump_base_experiment_code"]
