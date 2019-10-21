from collections import OrderedDict
from typing import Union, Dict


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
    callbacks: dict,
    moment: str
):
    """
    Sort callbacks by their order

    Args:
        callbacks (dict): callbacks to sort
        moment (str): one of ``start`` or ``end``

    Returns:
        (OrderedDict): sorted callbacks by their ordering
    """
    if moment is None:
        moment = "start"
    elif moment not in ["start", "end"]:
        raise ValueError(f"Got unknown value for moment: {moment}")

    result = sorted(
        callbacks.items(),
        key=lambda callback_kv: get_callback_orders(callback_kv[1])[moment]
    )

    result = OrderedDict(result)

    return result


def get_loggers(
    callbacks: dict,
    moment: str
):
    """
    Outputs a dict of loggers

    Args:
        callbacks (dict): all callbacks
        moment (str): one of ``start`` or ``end``

    Returns:
        (dict): only the loggers from the callbacks
    """
    if moment is None:
        moment = "start"
    elif moment not in ["start", "end"]:
        raise ValueError(f"Got unknown value for moment: {moment}")

    from ..core import CallbackOrder
    if moment == "start":
        logger_order = CallbackOrder.Logger_pre
    else:
        logger_order = CallbackOrder.Logger

    result = {
        key: value
        for key, value in callbacks.items()
        if get_callback_orders(value)[moment] == logger_order
    }

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
        result = callbacks
    elif isinstance(callbacks, list):
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


__all__ = [
    "get_callback_orders",
    "get_sorted_callbacks",
    "get_loggers",
    "process_callback"
]
