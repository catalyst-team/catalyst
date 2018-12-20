import os
import shutil
import pathlib
from datetime import datetime

from catalyst.utils.misc import import_module


def prepare_modules(model_dir, dump_dir=None):
    model_dir = model_dir[:-1] if model_dir.endswith("/") else model_dir
    model_dir_name = model_dir.rsplit("/", 1)[-1]

    if dump_dir is not None:
        current_date = datetime.now().strftime("%y-%m-%d-%H-%M-%S-%M-%f")
        new_src_dir = f"/src-{current_date}/"

        # @TODO: hardcoded
        old_pro_dir = os.path.dirname(os.path.abspath(__file__)) + "/../../"
        new_pro_dir = dump_dir + f"/{new_src_dir}/catalyst/"
        shutil.copytree(old_pro_dir, new_pro_dir)

        old_model_dir = os.path.abspath(model_dir)
        model_dir_ = model_dir.rsplit("/", 1)[-1]
        new_model_dir = dump_dir + f"/{new_src_dir}/{model_dir_}/"
        shutil.copytree(old_model_dir, new_model_dir)

    pyfiles = list(
        map(lambda x: x.name[:-3],
            pathlib.Path(model_dir).glob("*.py"))
    )

    modules = {}
    for name in pyfiles:
        module_name = f"{model_dir_name}.{name}"
        module_src = model_dir + "/" + f"{name}.py"

        module = import_module(module_name, module_src)
        modules[name] = module

    return modules
