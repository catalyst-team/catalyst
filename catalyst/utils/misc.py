from typing import Iterable, Any

import copy
import collections
import numpy as np
from itertools import tee
import shutil
from pathlib import Path


def pairwise(iterable: Iterable[Any]) -> Iterable[Any]:
    """
    Iterate sequences by pairs

    Args:
        iterable: Any iterable sequence

    Returns:
        pairwise iterator

    Examples:
        >>> for i in pairwise([1, 2, 5, -3]):
        >>>     print(i)
        (1, 2)
        (2, 5)
        (5, -3)
    """
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def make_tuple(tuple_like):
    tuple_like = (
        tuple_like if isinstance(tuple_like, (list, tuple)) else
        (tuple_like, tuple_like)
    )
    return tuple_like


def merge_dicts(*dicts: dict) -> dict:
    """
    Recursive dict merge.
    Instead of updating only top-level keys,
    ``merge_dicts`` recurses down into dicts nested
    to an arbitrary depth, updating keys.

    Args:
        *dicts: several dictionaries to merge

    Returns:
        dict: deep-merged dictionary
    """
    assert len(dicts) > 1

    dict_ = copy.deepcopy(dicts[0])

    for merge_dict in dicts[1:]:
        merge_dict = merge_dict or {}
        for k, v in merge_dict.items():
            if (
                k in dict_ and isinstance(dict_[k], dict)
                and isinstance(merge_dict[k], collections.Mapping)
            ):
                dict_[k] = merge_dicts(dict_[k], merge_dict[k])
            else:
                dict_[k] = merge_dict[k]

    return dict_


def append_dict(dict1, dict2):
    """
    Appends dict2 with the same keys as dict1 to dict1
    """
    for key in dict1.keys():
        dict1[key] = np.concatenate((dict1[key], dict2[key]))
    return dict1


def flatten_dict(d, parent_key="", sep="/"):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return collections.OrderedDict(items)


def is_exception(ex: Any) -> bool:
    """
    Check if the argument is of Exception type
    """
    result = (ex is not None) and isinstance(ex, BaseException)
    return result


def copy_directory(input_dir: Path, output_dir: Path) -> None:
    """
    Recursively copies the input directory

    Args:
        input_dir (Path): input directory
        output_dir (Path): output directory
    """
    output_dir.mkdir(exist_ok=True, parents=True)
    for path in input_dir.iterdir():
        if path.is_dir():
            path_name = path.name
            copy_directory(path, output_dir / path_name)
        else:
            shutil.copy2(path, output_dir)
