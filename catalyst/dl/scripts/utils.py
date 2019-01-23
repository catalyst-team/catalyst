import os
import shutil
import pathlib
from datetime import datetime

from catalyst.utils.misc import import_module


def prepare_modules(expdir, dump_dir=None):
    expdir = expdir[:-1] if expdir.endswith("/") else expdir
    expdir_name = expdir.rsplit("/", 1)[-1]

    if dump_dir is not None:
        current_date = datetime.now().strftime("%y-%m-%d-%H-%M-%S-%M-%f")
        new_src_dir = f"/src-{current_date}/"

        # @TODO: hardcoded
        old_pro_dir = os.path.dirname(os.path.abspath(__file__)) + "/../../"
        new_pro_dir = dump_dir + f"/{new_src_dir}/catalyst/"
        shutil.copytree(old_pro_dir, new_pro_dir)

        old_expdir = os.path.abspath(expdir)
        expdir_ = expdir.rsplit("/", 1)[-1]
        new_expdir = dump_dir + f"/{new_src_dir}/{expdir_}/"
        shutil.copytree(old_expdir, new_expdir)

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
