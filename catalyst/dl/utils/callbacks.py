from collections import OrderedDict
from typing import Union, List


def get_sorted_callbacks(callbacks: list, event: str = "start") -> list:
    """
    Sort callbacks by their order

    Args:
        callbacks (List[Callback]): callbacks to sort
        event (str): event name to sort callbacks ('start' or 'end')

    Returns:
        (List[Callback]): sorted callbacks
    """
    def key_to_sort(callback):
        if isinstance(callback.order, tuple):
            pre, post = callback.order
            if event == "start":
                return pre
            else:
                return post

        return callback.order

    result = sorted(callbacks, key=key_to_sort)

    return result


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
        # result = [(k, v) for k, v in callbacks.items()]
        # result = sorted(result, key=lambda x: x[1].order)
        # result = OrderedDict(result)
        return callbacks
    elif isinstance(callbacks, list):
        # result = sorted(callbacks, key=lambda x: x.order)
        result = OrderedDict([
            (i, value)
            for i, value in enumerate(callbacks)
        ])
    else:
        raise TypeError(
            f"Callbacks must be either OrderedDict or list, "
            f"got {type(callbacks)}"
        )

    return result


__all__ = ["process_callback"]
