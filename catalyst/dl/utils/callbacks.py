from catalyst.dl import Callback, WrapperCallback


def get_original_callback(callback: Callback) -> Callback:
    """Get original callback (if it has wrapper)

    Args:
        callback (Callback): callback to unpack

    Returns:
        callback inside wrapper
    """
    while isinstance(callback, WrapperCallback):
        callback = callback.callback
    return callback


def check_callback_isinstance(callback: Callback, class_or_tuple) -> bool:
    """Check if first callback is the same type as second callback

    Args:
        callback (Callback): callback to check
        second (Callback): callback onject to compare with

    Returns:
        bool: true if first object has the same type as second
    """
    callback = get_original_callback(callback)
    return isinstance(callback, class_or_tuple)


__all__ = ["get_original_callback", "check_callback_isinstance"]
