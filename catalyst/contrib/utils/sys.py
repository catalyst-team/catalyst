from typing import Union
from importlib.util import module_from_spec, spec_from_file_location
import os
from pathlib import Path
import sys


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


__all__ = ["import_module"]
