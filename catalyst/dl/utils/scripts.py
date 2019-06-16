import pathlib
from catalyst.utils.scripts import import_module


def import_experiment_and_runner(expdir: pathlib.Path):
    if not isinstance(expdir, pathlib.Path):
        expdir = pathlib.Path(expdir)
    m = import_module(expdir)
    Experiment, Runner = m.Experiment, m.Runner
    return Experiment, Runner


__all__ = ["import_experiment_and_runner"]
