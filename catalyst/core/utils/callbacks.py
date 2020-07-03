from typing import Dict, List, Union
from collections import OrderedDict

from catalyst.core.callback import CallbackNode
from catalyst.core.utils import get_rank


def sort_callbacks_by_order(
    callbacks: Union[List, Dict, OrderedDict]
) -> OrderedDict:
    """Creates an sequence of callbacks and sort them.

    Args:
        callbacks: either list of callbacks or ordered dict

    Returns:
        sequence of callbacks sorted by ``callback order``

    Raises:
        TypeError: if `callbacks` is out of
            `None`, `dict`, `OrderedDict`, `list`
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


def filter_callbacks_by_node(
    callbacks: Union[Dict, OrderedDict]
) -> Union[Dict, OrderedDict]:
    """
    Filters callbacks based on running node.
    Deletes worker-only callbacks from ``CallbackNode.Master``
    and master-only callbacks from ``CallbackNode.Worker``.

    Args:
        callbacks (Union[Dict, OrderedDict]): callbacks

    Returns:
        Union[Dict, OrderedDict]: filtered callbacks dictionary.
    """
    # distributed run setting
    output = callbacks.copy()
    rank = get_rank()
    if rank == 0:  # master node
        # remove worker-only callbacks on master node
        for k in list(
            filter(lambda c: output[c].node == CallbackNode.worker, output)
        ):
            del output[k]
    elif rank > 0:  # worker node
        # remove master-only callbacks on worker nodes
        for k in list(
            filter(lambda c: output[c].node == CallbackNode.master, output)
        ):
            del output[k]
    return output


__all__ = ["sort_callbacks_by_order", "filter_callbacks_by_node"]
