from typing import Dict, List, Union
from collections import OrderedDict
from copy import copy
import warnings

from torch.utils.data import DataLoader, DistributedSampler

from catalyst.core.callback import Callback, CallbackNode, CallbackWrapper
from catalyst.data.sampler import DistributedSamplerWrapper
from catalyst.utils.distributed import get_rank


def _force_make_distributed_loader(loader: DataLoader) -> DataLoader:
    """
    Transfers loader to distributed mode. Experimental feature.

    Args:
        loader: pytorch dataloder

    Returns:
        DataLoader: pytorch dataloder with distributed sampler.
    """
    sampler = (
        DistributedSampler(dataset=loader.dataset)
        if getattr(loader, "sampler", None) is not None
        else DistributedSamplerWrapper(sampler=loader.sampler)
    )
    loader = DataLoader(
        dataset=copy(loader.dataset),
        batch_size=loader.batch_size,
        # shuffle=loader.shuffle,
        sampler=sampler,
        # batch_sampler=loader.batch_sampler,
        num_workers=loader.num_workers,
        # collate_fn=loader.collate_fn,
        pin_memory=loader.pin_memory,
        drop_last=loader.drop_last,
    )
    return loader


def validate_loaders(loaders: Dict[str, DataLoader]) -> Dict[str, DataLoader]:
    """
    Check pytorch dataloaders for distributed setup.
    Transfers them to distributed mode if necessary.
    (Experimental feature)

    Args:
        loaders: dictionary with pytorch dataloaders

    Returns:
        Dict[str, DataLoader]: dictionary
            with pytorch dataloaders (with distributed samplers if necessary)
    """
    rank = get_rank()
    if rank >= 0:
        for key, value in loaders.items():
            if not isinstance(value.sampler, (DistributedSampler, DistributedSamplerWrapper)):
                warnings.warn(
                    "With distributed training setup, "
                    "you need ``DistributedSampler`` for your ``DataLoader``."
                    "Transferring to distributed mode. (Experimental feature)"
                )
                loaders[key] = _force_make_distributed_loader(value)
    return loaders


def _get_original_callback(callback: Callback) -> Callback:
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
    callback = _get_original_callback(callback)
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
            f"Callbacks must be either Dict/OrderedDict or list, " f"got {type(callbacks)}"
        )

    return output


def filter_callbacks_by_node(callbacks: Union[Dict, OrderedDict]) -> Union[Dict, OrderedDict]:
    """
    Filters callbacks based on running node.
    Deletes worker-only callbacks from ``CallbackNode.Master``
    and master-only callbacks from ``CallbackNode.Worker``.

    Args:
        callbacks: callbacks

    Returns:
        Union: filtered callbacks dictionary.
    """
    # distributed run setting
    output = callbacks.copy()
    rank = get_rank()
    if rank == 0:  # master node
        # remove worker-only callbacks on master node
        for k in list(filter(lambda c: output[c].node == CallbackNode.worker, output)):
            del output[k]
    elif rank > 0:  # worker node
        # remove master-only callbacks on worker nodes
        for k in list(filter(lambda c: output[c].node == CallbackNode.master, output)):
            del output[k]
    return output


__all__ = [
    "validate_loaders",
    "sort_callbacks_by_order",
    "filter_callbacks_by_node",
    "callback_isinstance",
]
