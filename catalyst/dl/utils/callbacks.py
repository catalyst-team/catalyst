from collections import OrderedDict
from typing import Union


def process_callback(
    callbacks: Union[list, OrderedDict]
) -> OrderedDict:
    """
    Creates an sequence of callbacks and sort them
    Args:
        callbacks: either list of callbacks or ordered dict

    Returns:
        sequence of callbacks sorted by ``callback order``
    """
    if callbacks is None:
        result = OrderedDict()
    elif isinstance(callbacks, OrderedDict):
        result = [(k, v) for k, v in callbacks.items()]
        result = sorted(result, key=lambda x: x[1].order)
        result = OrderedDict(result)
    elif isinstance(callbacks, list):
        result = sorted(callbacks, key=lambda x: x.order)
        result = OrderedDict([
            (i, value)
            for i, value in enumerate(result)
        ])
    else:
        raise TypeError(
            f"Callbacks must be either OrderedDict or list, "
            f"got {type(callbacks)}"
        )

    return result


__all__ = ["process_callback"]
