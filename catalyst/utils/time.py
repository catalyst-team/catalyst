from datetime import datetime


def get_utcnow_time(format: str = None) -> str:
    """
    Return string with current utc time in chosen format

    Args:
        format (str): format string. if None "%y%m%d.%H%M%S" will be used.

    Returns:
        str: formatted utc time string
    """
    if format is None:
        format = "%y%m%d.%H%M%S"
    result = datetime.utcnow().strftime(format)
    return result


__all__ = ["get_utcnow_time"]
