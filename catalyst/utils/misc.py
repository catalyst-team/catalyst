from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, TypeVar, Union
import argparse
from base64 import urlsafe_b64encode
import collections
import copy
from datetime import datetime
from hashlib import sha256
import inspect
from itertools import tee
import random

import numpy as np
from torch import int as tint, long, short, Tensor

T = TypeVar("T")


def boolean_flag(
    parser: argparse.ArgumentParser,
    name: str,
    default: Optional[bool] = False,
    help: str = None,  # noqa: WPS125
    shorthand: str = None,
) -> None:
    """Add a boolean flag to a parser inplace.

    Examples:
        >>> parser = argparse.ArgumentParser()
        >>> boolean_flag(
        >>>     parser, "flag", default=False, help="some flag", shorthand="f"
        >>> )

    Args:
        parser: parser to add the flag to
        name: argument name
            --<name> will enable the flag,
            while --no-<name> will disable it
        default (bool, optional): default value of the flag
        help: help string for the flag
        shorthand: shorthand string for the argument
    """
    dest = name.replace("-", "_")
    names = ["--" + name]
    if shorthand is not None:
        names.append("-" + shorthand)
    parser.add_argument(*names, action="store_true", default=default, dest=dest, help=help)
    parser.add_argument("--no-" + name, action="store_false", dest=dest)


def set_global_seed(seed: int) -> None:
    """Sets random seed into Numpy and Random, PyTorch and TensorFlow.

    Args:
        seed: random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        pass
    else:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)


def maybe_recursive_call(
    object_or_dict,
    method: Union[str, Callable],
    recursive_args=None,
    recursive_kwargs=None,
    **kwargs,
):
    """Calls the ``method`` recursively for the ``object_or_dict``.

    Args:
        object_or_dict: some object or a dictionary of objects
        method: method name to call
        recursive_args: list of arguments to pass to the ``method``
        recursive_kwargs: list of key-arguments to pass to the ``method``
        **kwargs: Arbitrary keyword arguments

    Returns:
        result of `method` call
    """
    if isinstance(object_or_dict, dict):
        result = type(object_or_dict)()
        for k, v in object_or_dict.items():
            r_args = None if recursive_args is None else recursive_args[k]
            r_kwargs = None if recursive_kwargs is None else recursive_kwargs[k]
            result[k] = maybe_recursive_call(
                v, method, recursive_args=r_args, recursive_kwargs=r_kwargs, **kwargs,
            )
        return result

    r_args = recursive_args or []
    if not isinstance(r_args, (list, tuple)):
        r_args = [r_args]
    r_kwargs = recursive_kwargs or {}
    if isinstance(method, str):
        return getattr(object_or_dict, method)(*r_args, **r_kwargs, **kwargs)
    else:
        return method(object_or_dict, *r_args, **r_kwargs, **kwargs)


def is_exception(ex: Any) -> bool:
    """Check if the argument is of ``Exception`` type."""
    result = (ex is not None) and isinstance(ex, BaseException)
    return result


def get_utcnow_time(format: str = None) -> str:
    """Return string with current utc time in chosen format.

    Args:
        format: format string. if None "%y%m%d.%H%M%S" will be used.

    Returns:
        str: formatted utc time string
    """
    if format is None:
        format = "%y%m%d.%H%M%S"
    result = datetime.utcnow().strftime(format)
    return result


def get_fn_default_params(fn: Callable[..., Any], exclude: List[str] = None):
    """Return default parameters of Callable.

    Args:
        fn (Callable[..., Any]): target Callable
        exclude: exclude list of parameters

    Returns:
        dict: contains default parameters of `fn`
    """
    argspec = inspect.getfullargspec(fn)
    default_params = zip(argspec.args[-len(argspec.defaults) :], argspec.defaults)
    if exclude is not None:
        default_params = filter(lambda x: x[0] not in exclude, default_params)
    default_params = dict(default_params)
    return default_params


def get_fn_argsnames(fn: Callable[..., Any], exclude: List[str] = None):
    """Return parameter names of Callable.

    Args:
        fn (Callable[..., Any]): target Callable
        exclude: exclude list of parameters

    Returns:
        list: contains parameter names of `fn`
    """
    argspec = inspect.getfullargspec(fn)
    params = argspec.args + argspec.kwonlyargs
    if exclude is not None:
        params = list(filter(lambda x: x not in exclude, params))
    return params


def get_attr(obj: Any, key: str, inner_key: str = None) -> Any:
    """
    Alias for python `getattr` method. Useful for Callbacks preparation
    and cases with multi-criterion, multi-optimizer setup.
    For example, when you would like to train multi-task classification.

    Used to get a named attribute from a `IRunner` by `key` keyword;
    for example\
    ::

        get_attr(runner, "criterion")
        # is equivalent to
        runner.criterion

        get_attr(runner, "optimizer")
        # is equivalent to
        runner.optimizer

        get_attr(runner, "scheduler")
        # is equivalent to
        runner.scheduler

    With `inner_key` usage, it suppose to find a dictionary under `key`\
    and would get `inner_key` from this dict; for example,
    ::

        get_attr(runner, "criterion", "bce")
        # is equivalent to
        runner.criterion["bce"]

        get_attr(runner, "optimizer", "adam")
        # is equivalent to
        runner.optimizer["adam"]

        get_attr(runner, "scheduler", "adam")
        # is equivalent to
        runner.scheduler["adam"]

    Args:
        obj: object of interest
        key: name for attribute of interest,
            like `criterion`, `optimizer`, `scheduler`
        inner_key: name of inner dictionary key

    Returns:
        inner attribute
    """
    if inner_key is None:
        return getattr(obj, key)
    else:
        return getattr(obj, key)[inner_key]


def _get_key_str(dictionary: dict, key: Optional[Union[str, List[str]]]) -> Any:
    return dictionary[key]


def _get_key_list(dictionary: dict, key: Optional[Union[str, List[str]]]) -> Dict:
    result = {name: dictionary[name] for name in key}
    return result


def _get_key_dict(dictionary: dict, key: Optional[Union[str, List[str]]]) -> Dict:
    result = {key_out: dictionary[key_in] for key_in, key_out in key.items()}
    return result


def _get_key_none(dictionary: dict, key: Optional[Union[str, List[str]]]) -> Dict:
    return {}


def _get_key_all(dictionary: dict, key: Optional[Union[str, List[str]]]) -> Dict:
    return dictionary


def get_dictkey_auto_fn(key: Optional[Union[str, List[str]]]) -> Callable:
    """Function generator for sub-dict preparation from dict based on predefined keys.

    Args:
        key: keys

    Returns:
        function

    Raises:
        NotImplementedError: if key is out of
            `str`, `tuple`, `list`, `dict`, `None`
    """
    if isinstance(key, str):
        if key == "__all__":
            return _get_key_all
        else:
            return _get_key_str
    elif isinstance(key, (list, tuple)):
        return _get_key_list
    elif isinstance(key, dict):
        return _get_key_dict
    elif key is None:
        return _get_key_none
    else:
        raise NotImplementedError()


def merge_dicts(*dicts: dict) -> dict:
    """Recursive dict merge.
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
        for k in merge_dict:
            if (
                k in dict_
                and isinstance(dict_[k], dict)
                and isinstance(merge_dict[k], collections.Mapping)
            ):
                dict_[k] = merge_dicts(dict_[k], merge_dict[k])
            else:
                dict_[k] = merge_dict[k]

    return dict_


def flatten_dict(
    dictionary: Dict[str, Any], parent_key: str = "", separator: str = "/"
) -> "collections.OrderedDict":
    """Make the given dictionary flatten.

    Args:
        dictionary: giving dictionary
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
            items.extend(flatten_dict(value, new_key, separator=separator).items())
        else:
            items.append((new_key, value))
    return collections.OrderedDict(items)


def _make_hashable(o):
    if isinstance(o, (tuple, list)):
        return tuple(((type(o).__name__, _make_hashable(e)) for e in o))
    if isinstance(o, dict):
        return tuple(sorted((type(o).__name__, k, _make_hashable(v)) for k, v in o.items()))
    if isinstance(o, (set, frozenset)):
        return tuple(sorted((type(o).__name__, _make_hashable(e)) for e in o))
    return o


def get_hash(obj: Any) -> str:
    """
    Creates unique hash from object following way:
    - Represent obj as sting recursively
    - Hash this string with sha256 hash function
    - encode hash with url-safe base64 encoding

    Args:
        obj: object to hash

    Returns:
        base64-encoded string
    """
    bytes_to_hash = repr(_make_hashable(obj)).encode()
    hash_bytes = sha256(bytes_to_hash).digest()
    return urlsafe_b64encode(hash_bytes).decode()


def get_short_hash(obj) -> str:
    """
    Creates unique short hash from object.

    Args:
        obj: object to hash

    Returns:
        short base64-encoded string (6 chars)
    """
    hash_ = get_hash(obj)[:6]
    return hash_


def pairwise(iterable: Iterable[T]) -> Iterable[Tuple[T, T]]:
    """Iterate sequences by pairs.

    Examples:
        >>> for i in pairwise([1, 2, 5, -3]):
        >>>     print(i)
        (1, 2)
        (2, 5)
        (5, -3)

    Args:
        iterable: Any iterable sequence

    Returns:
        pairwise iterator
    """
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def make_tuple(tuple_like):
    """Creates a tuple if given ``tuple_like`` value isn't list or tuple.

    Args:
        tuple_like: tuple like object - list or tuple

    Returns:
        tuple or list
    """
    tuple_like = tuple_like if isinstance(tuple_like, (list, tuple)) else (tuple_like, tuple_like)
    return tuple_like


def args_are_not_none(*args: Optional[Any]) -> bool:
    """Check that all arguments are not ``None``.

    Args:
        *args: values  # noqa: RST213

    Returns:
         bool: True if all value were not None, False otherwise
    """
    if args is None:
        return False

    for arg in args:
        if arg is None:
            return False

    return True


def find_value_ids(it: Iterable[Any], value: Any) -> List[int]:
    """
    Args:
        it: list of any
        value: query element

    Returns:
        indices of the all elements equal x0
    """
    if isinstance(it, np.ndarray):
        inds = list(np.where(it == value)[0])
    else:  # could be very slow
        inds = [i for i, el in enumerate(it) if el == value]
    return inds


def get_by_keys(dict_: dict, *keys: Any, default: Optional[T] = None) -> T:
    """@TODO: docs."""
    if not isinstance(dict_, dict):
        raise ValueError()

    key, *keys = keys
    if len(keys) == 0 or key not in dict_:
        return dict_.get(key, default)
    return get_by_keys(dict_[key], *keys, default=default)


def convert_labels2list(labels: Union[Tensor, List[int]]) -> List[int]:
    """
    This function allows to work with 2 types of indexing:
    using a integer tensor and a list of indices.

    Args:
        labels: labels of batch samples

    Returns:
        labels of batch samples in the aligned format

    Raises:
        TypeError: if type of input labels is not tensor and list
    """
    if isinstance(labels, Tensor):
        labels = labels.squeeze()
        assert (len(labels.shape) == 1) and (
            labels.dtype in [short, tint, long]
        ), "Labels cannot be interpreted as indices."
        labels_list = labels.tolist()
    elif isinstance(labels, list):
        labels_list = labels.copy()
    else:
        raise TypeError(f"Unexpected type of labels: {type(labels)}).")

    return labels_list


__all__ = [
    "boolean_flag",
    "get_fn_default_params",
    "get_fn_argsnames",
    "get_utcnow_time",
    "is_exception",
    "maybe_recursive_call",
    "get_attr",
    "set_global_seed",
    "get_dictkey_auto_fn",
    "merge_dicts",
    "flatten_dict",
    "get_hash",
    "get_short_hash",
    "args_are_not_none",
    "make_tuple",
    "pairwise",
    "find_value_ids",
    "get_by_keys",
    "convert_labels2list",
]
