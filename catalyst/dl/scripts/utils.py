import os
import sys
import shutil
import pathlib
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
    m = import_module(expdir)
    Experiment, Runner = m.Experiment, m.Runner
    return Experiment, Runner


def _tricky_dir_copy(dir_from, dir_to):
    os.makedirs(dir_to, exist_ok=True)
    shutil.rmtree(dir_to)
    shutil.copytree(dir_from, dir_to)


def dump_code(expdir, logdir):
    expdir = expdir[:-1] if expdir.endswith("/") else expdir
    new_src_dir = f"/code/"

    # @TODO: hardcoded
    old_pro_dir = os.path.dirname(os.path.abspath(__file__)) + "/../../"
    new_pro_dir = logdir + f"/{new_src_dir}/catalyst/"
    _tricky_dir_copy(old_pro_dir, new_pro_dir)

    old_expdir = os.path.abspath(expdir)
    expdir_ = expdir.rsplit("/", 1)[-1]
    new_expdir = logdir + f"/{new_src_dir}/{expdir_}/"
    _tricky_dir_copy(old_expdir, new_expdir)


def prepare_modules(expdir, dump_dir=None):
    expdir = expdir[:-1] if expdir.endswith("/") else expdir
    expdir_name = expdir.rsplit("/", 1)[-1]

    if dump_dir is not None:
        dump_code(expdir, dump_dir)

    pyfiles = list(
        map(lambda x: x.name[:-3],
            pathlib.Path(expdir).glob("*.py"))
    )

    modules = {}
    for name in pyfiles:
        module_name = f"{expdir_name}.{name}"
        module_src = expdir + "/" + f"{name}.py"

        module = import_module(module_name, module_src)
        modules[name] = module

    return modules
