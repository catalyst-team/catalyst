from typing import Dict, List, Tuple, Union
from collections import OrderedDict
from functools import lru_cache
import warnings

from torch.utils.data import BatchSampler, DataLoader

from catalyst.core.callback import (
    Callback,
    CallbackWrapper,
    IBackwardCallback,
    ICriterionCallback,
    IOptimizerCallback,
    ISchedulerCallback,
)
from catalyst.typing import RunnerCriterion, RunnerOptimizer, RunnerScheduler


def get_original_callback(callback: Callback) -> Callback:
    """Docs."""
    while isinstance(callback, CallbackWrapper):
        callback = callback.callback
    return callback


def callback_isinstance(callback: Callback, class_or_tuple) -> bool:
    """Check if callback is the same type as required ``class_or_tuple``

    Args:
        callback: callback to check
        class_or_tuple: class_or_tuple to compare with

    Returns:
        bool: true if first object has the required type
    """
    callback = get_original_callback(callback)
    return isinstance(callback, class_or_tuple)


def sort_callbacks_by_order(
    callbacks: Union[List, Dict, OrderedDict]
) -> "OrderedDict[str, Callback]":
    """Creates an sequence of callbacks and sort them.

    Args:
        callbacks: either list of callbacks or ordered dict

    Returns:
        sequence of callbacks sorted by ``callback order``

    Raises:
        TypeError: if `callbacks` is out of `None`, `dict`, `OrderedDict`, `list`
    """
    if callbacks is None:
        output = OrderedDict()
    elif isinstance(callbacks, (dict, OrderedDict)):
        output = [(k, v) for k, v in callbacks.items()]
        output = sorted(output, key=lambda x: x[1].order)
        output = OrderedDict(output)
    elif isinstance(callbacks, list):
        output = sorted(callbacks, key=lambda x: x.order)
        output = OrderedDict([(i, value) for i, value in enumerate(output)])
    else:
        raise TypeError(
            f"Callbacks must be either Dict/OrderedDict or list, "
            f"got {type(callbacks)}"
        )

    return output


@lru_cache(maxsize=42)
def is_str_intersections(origin_string: str, strings: Tuple):
    """Docs."""
    return any(x in origin_string for x in strings)


def get_loader_batch_size(loader: DataLoader):
    """Docs."""
    batch_size = loader.batch_size
    if batch_size is not None:
        return batch_size

    batch_size = loader.batch_sampler.batch_size
    if batch_size is not None:
        return batch_size
    raise NotImplementedError(
        "No `batch_size` found,"
        "please specify it with `loader.batch_size`,"
        "or `loader.batch_sampler.batch_size`"
    )


def get_loader_num_samples(loader: DataLoader):
    """Docs."""
    batch_size = get_loader_batch_size(loader)
    if isinstance(loader.batch_sampler, BatchSampler):
        # pytorch default item-based samplers
        if loader.drop_last:
            return (len(loader.dataset) // batch_size) * batch_size
        else:
            return len(loader.dataset)
    else:
        # pytorch batch-based samplers
        return len(loader) * batch_size


def check_callbacks(
    callbacks: OrderedDict,
    criterion: RunnerCriterion = None,
    optimizer: RunnerOptimizer = None,
    scheduler: RunnerScheduler = None,
):
    """Docs."""
    callback_exists = lambda callback_fn: any(
        callback_isinstance(x, callback_fn) for x in callbacks.values()
    )
    if criterion is not None and not callback_exists(ICriterionCallback):
        warnings.warn(
            "No ``ICriterionCallback/CriterionCallback`` were found "
            "while runner.criterion is not None."
            "Do you compute the loss during ``runner.handle_batch``?"
        )
    if (criterion is not None or optimizer is not None) and not callback_exists(
        IBackwardCallback
    ):
        warnings.warn(
            "No ``IBackwardCallback/BackwardCallback`` were found "
            "while runner.criterion/optimizer is not None."
            "Do you backward the loss during ``runner.handle_batch``?"
        )
    if optimizer is not None and not callback_exists(IOptimizerCallback):
        warnings.warn(
            "No ``IOptimizerCallback/OptimizerCallback`` were found "
            "while runner.optimizer is not None."
            "Do run optimisation step pass during ``runner.handle_batch``?"
        )
    if scheduler is not None and not callback_exists(ISchedulerCallback):
        warnings.warn(
            "No ``ISchedulerCallback/SchedulerCallback`` were found "
            "while runner.scheduler is not None."
            "Do you make scheduler step during ``runner.handle_batch``?"
        )


__all__ = [
    "get_original_callback",
    "callback_isinstance",
    "check_callbacks",
    "is_str_intersections",
    "get_loader_batch_size",
    "get_loader_num_samples",
    "sort_callbacks_by_order",
]
