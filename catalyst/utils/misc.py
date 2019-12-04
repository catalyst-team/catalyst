from typing import Any, Iterable, Optional, Dict  # isort:skip

import collections
import copy
from datetime import datetime
from itertools import tee
from pathlib import Path
import shutil

import numpy as np


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
    """
    Creates a tuple if given ``tuple_like`` value isn't list or tuple

    Returns:
        tuple or list
    """
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


def flatten_dict(
    dictionary: Dict[str, Any],
    parent_key: str = "",
    separator: str = "/"
) -> "collections.OrderedDict":
    """
    Make the given dictionary flatten

    Args:
        dictionary (dict): giving dictionary
        parent_key (str, optional): prefix nested keys with
            string ``parent_key``
        separator (str, optional): delimiter between
            ``parent_key`` and ``key`` to use

    Returns:
        collections.OrderedDict: ordered dictionary with flatten keys
    """
    items = []
    for key, value in dictionary.items():
        new_key = parent_key + separator + key if parent_key else key
        if isinstance(value, collections.MutableMapping):
            items.extend(
                flatten_dict(value, new_key, separator=separator).items()
            )
        else:
            items.append((new_key, value))
    return collections.OrderedDict(items)


def maybe_recursive_call(
    object_or_dict,
    method: str,
    recursive_args=None,
    recursive_kwargs=None,
    **kwargs,
):
    """
    Calls the ``method`` recursively for the object_or_dict

    Args:
        object_or_dict (Any): some object or a dictinary of objects
        method (str): method name to call
        recursive_args: list of arguments to pass to the ``method``
        recursive_kwargs: list of key-arguments to pass to the ``method``
        **kwargs: Arbitrary keyword arguments
    """
    if isinstance(object_or_dict, dict):
        result = type(object_or_dict)()
        for k, v in object_or_dict.items():
            r_args = \
                None if recursive_args is None else recursive_args[k]
            r_kwargs = \
                None if recursive_kwargs is None else recursive_kwargs[k]
            result[k] = maybe_recursive_call(
                v,
                method,
                recursive_args=r_args,
                recursive_kwargs=r_kwargs,
                **kwargs,
            )
        return result

    r_args = recursive_args or []
    if not isinstance(r_args, (list, tuple)):
        r_args = [r_args]
    r_kwargs = recursive_kwargs or {}
    return getattr(object_or_dict, method)(*r_args, **r_kwargs, **kwargs)


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


def get_utcnow_time(format: str = None) -> str:
    """
    Return string with current utc time in chosen format

    Args:
        format (str): format string. if None "%y%m%d.%H%M%S" will be used.

    Returns:
        str: formatted utc time string
    """
    if format is None:
        format = "%y%m%d.%H%M%S"
    result = datetime.utcnow().strftime(format)
    return result


def format_metric(name: str, value: float) -> str:
    """
    Format metric. Metric will be returned in the scientific format if 4
    decimal chars are not enough (metric value lower than 1e-4)

    Args:
        name (str): metric name
        value (float): value of metric
    """
    if value < 1e-4:
        return f"{name}={value:1.3e}"
    return f"{name}={value:.4f}"


def args_are_not_none(*args: Optional[Any]) -> bool:
    """
    Check that all arguments are not None
    Args:
        *args (Any): values
    Returns:
         bool: True if all value were not None, False otherwise
    """
    if args is None:
        return False

    for arg in args:
        if arg is None:
            return False

    return True
