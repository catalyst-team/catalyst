from collections import OrderedDict
from typing import Union, List, Tuple


def get_callback_orders(callback) -> Tuple[int, int, int]:
    order = callback.order
    if isinstance(order, tuple):
        if len(order) == 2:
            pre, post = order
            middle = None
        else:
            pre, middle, post = order
    else:
        middle = order
        pre, post = None, None

    return pre, middle, post


def _sorted_callbacks(callbacks):
    result = [
        callback
        for callback, order in sorted(callbacks, key=lambda x: x[1])
    ]

    return result


def split_sorted_callbacks(
    callbacks: list
):
    """
    Sort callbacks by their order

    Args:
        callbacks (List[Callback]): callbacks to sort
    """
    pre_callbacks = []
    middle_callbacks = []
    post_callbacks = []

    for callback in callbacks:
        pre, middle, post = get_callback_orders(callback)

        if pre is not None:
            pre_callbacks.append((callback, pre))
        if middle is not None:
            middle_callbacks.append((callback, middle))
        if post is not None:
            post_callbacks.append((callback, post))

    pre_callbacks = _sorted_callbacks(pre_callbacks)
    middle_callbacks = _sorted_callbacks(middle_callbacks)
    post_callbacks = _sorted_callbacks(post_callbacks)

    return pre_callbacks, middle_callbacks, post_callbacks


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
