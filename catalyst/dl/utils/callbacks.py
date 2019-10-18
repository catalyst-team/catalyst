from collections import OrderedDict
from typing import Union, List, Tuple, Dict


def get_callback_orders(callback) -> Dict[str, int]:
    """
    Function that returns the callback orders for ``start`` and ``end``.

    Args:
        callback (Callback): input callback
    """
    order = callback.order

    result = {}
    if isinstance(order, dict):
        result = order
    elif isinstance(order, int):
        result["start"], result["end"] = order, order
    else:
        raise TypeError(f"Got invalid type for callback order: {type(order)}")

    return result


def get_sorted_callbacks(
    callbacks: list,
    moment: str
):
    """
    Sort callbacks by their order

    Args:
        callbacks (List[Callback]): callbacks to sort
        moment (str): one of ``start`` or ``end``
    """
    if moment is None:
        moment = "start"
    elif moment not in ["start", "end"]:
        raise ValueError(f"Got unknown value for moment: {moment}")

    result = sorted(
        callbacks,
        key=lambda callback: get_callback_orders(callback)[moment]
    )

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


__all__ = ["get_callback_orders", "get_sorted_callbacks", "process_callback"]
