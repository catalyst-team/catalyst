import os
import sys
import shutil
import pathlib
from datetime import datetime
from importlib.util import spec_from_file_location, module_from_spec


def import_module(expdir: pathlib.Path):
    # @TODO: better PYTHONPATH handling
    sys.path.insert(0, str(expdir.absolute()))
    sys.path.insert(0, os.path.dirname(str(expdir.absolute())))
    s = spec_from_file_location(
        expdir.name,
        str(expdir.absolute() / "__init__.py"),
        submodule_search_locations=[expdir.absolute()]
    )
    m = module_from_spec(s)
    s.loader.exec_module(m)
    sys.modules[expdir.name] = m
    return m


def import_experiment_and_runner(expdir: pathlib.Path):
    m = import_experiment_and_runner(expdir)
    Experiment, Runner = m.Experiment, m.Runner
    return Experiment, Runner


def dump_code(expdir, logdir):
    expdir = expdir[:-1] if expdir.endswith("/") else expdir

    current_date = datetime.now().strftime("%y-%m-%d-%H-%M-%S-%M-%f")
    new_src_dir = f"/src-{current_date}/"

    # @TODO: hardcoded
    old_pro_dir = os.path.dirname(os.path.abspath(__file__)) + "/../../"
    new_pro_dir = logdir + f"/{new_src_dir}/catalyst/"
    shutil.copytree(old_pro_dir, new_pro_dir)

    old_expdir = os.path.abspath(expdir)
    expdir_ = expdir.rsplit("/", 1)[-1]
    new_expdir = logdir + f"/{new_src_dir}/{expdir_}/"
    shutil.copytree(old_expdir, new_expdir)
