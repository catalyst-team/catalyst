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
    """Check if callback is the same type as required ``class_or_tuple``

    Args:
        callback (Callback): callback to check
        class_or_tuple: class_or_tuple to compare with

    Returns:
        bool: true if first object has the required type
    """
    callback = get_original_callback(callback)
    return isinstance(callback, class_or_tuple)


__all__ = ["get_original_callback", "check_callback_isinstance"]
