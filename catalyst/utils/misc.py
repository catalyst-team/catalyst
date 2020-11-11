from typing import Any, Callable, List, Union
from datetime import datetime
import inspect
from pathlib import Path
import random
import shutil

import numpy as np
from packaging.version import parse, Version


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
            torch.cuda.manual_seed_all(seed)
    try:
        import tensorflow as tf
    except ImportError:
        pass
    else:
        if parse(tf.__version__) >= Version("2.0.0"):
            tf.random.set_seed(seed)
        elif parse(tf.__version__) <= Version("1.13.2"):
            tf.set_random_seed(seed)
        else:
            tf.compat.v1.set_random_seed(seed)


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
            r_kwargs = (
                None if recursive_kwargs is None else recursive_kwargs[k]
            )
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
    if isinstance(method, str):
        return getattr(object_or_dict, method)(*r_args, **r_kwargs, **kwargs)
    else:
        return method(object_or_dict, *r_args, **r_kwargs, **kwargs)


def is_exception(ex: Any) -> bool:
    """Check if the argument is of ``Exception`` type."""
    result = (ex is not None) and isinstance(ex, BaseException)
    return result


def copy_directory(input_dir: Path, output_dir: Path) -> None:
    """Recursively copies the input directory.

    Args:
        input_dir: input directory
        output_dir: output directory
    """
    output_dir.mkdir(exist_ok=True, parents=True)
    for path in input_dir.iterdir():
        if path.is_dir():
            path_name = path.name
            copy_directory(path, output_dir / path_name)
        else:
            shutil.copy2(path, output_dir)


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


def format_metric(name: str, value: float) -> str:
    """Format metric.

    Metric will be returned in the scientific format if 4
    decimal chars are not enough (metric value lower than 1e-4).

    Args:
        name: metric name
        value: value of metric

    Returns:
        str: formatted metric
    """
    if value < 1e-4:
        return f"{name}={value:1.3e}"
    return f"{name}={value:.4f}"


def get_fn_default_params(fn: Callable[..., Any], exclude: List[str] = None):
    """Return default parameters of Callable.

    Args:
        fn (Callable[..., Any]): target Callable
        exclude: exclude list of parameters

    Returns:
        dict: contains default parameters of `fn`
    """
    argspec = inspect.getfullargspec(fn)
    default_params = zip(
        argspec.args[-len(argspec.defaults) :], argspec.defaults
    )
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

        # example 1
        runner.get_attr("criterion")
        # is equivalent to
        runner.criterion

        # example 2
        runner.get_attr("optimizer")
        # is equivalent to
        runner.optimizer

        # example 3
        runner.get_attr("scheduler")
        # is equivalent to
        runner.scheduler

    With `inner_key` usage, it suppose to find a dictionary under `key`\
    and would get `inner_key` from this dict; for example,
    ::

        # example 1
        runner.get_attr("criterion", "bce")
        # is equivalent to
        runner.criterion["bce"]

        # example 2
        runner.get_attr("optimizer", "adam")
        # is equivalent to
        runner.optimizer["adam"]

        # example 3
        runner.get_attr("scheduler", "adam")
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


__all__ = [
    "copy_directory",
    "format_metric",
    "get_fn_default_params",
    "get_fn_argsnames",
    "get_utcnow_time",
    "is_exception",
    "maybe_recursive_call",
    "get_attr",
    "set_global_seed",
]
