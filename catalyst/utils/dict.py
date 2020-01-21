from typing import Any, Callable, Dict, List, Optional, Union  # isort:skip


def get_key_str(
    dictionary: dict,
    key: Optional[Union[str, List[str]]],
) -> Any:
    """
    Takes value from dict by key.
    Args:
        dictionary: dict
        key: key

    Returns:
        value
    """
    return dictionary[key]


def get_key_list(
    dictionary: dict,
    key: Optional[Union[str, List[str]]],
) -> Dict:
    """
    Takes sub-dict from dict by list of keys.
    Args:
        dictionary: dict
        key: list of keys

    Returns:
        sub-dict
    """
    result = {key_: dictionary[key_] for key_ in key}
    return result


def get_key_dict(
    dictionary: dict,
    key: Optional[Union[str, List[str]]],
) -> Dict:
    """
    Takes sub-dict from dict by dict-mapping of keys.
    Args:
        dictionary: dict
        key: dict-mapping of keys

    Returns:
        sub-dict
    """
    result = {key_out: dictionary[key_in] for key_in, key_out in key.items()}
    return result


def get_key_none(
    dictionary: dict,
    key: Optional[Union[str, List[str]]],
) -> Dict:
    """
    Takes whole dict.
    Args:
        dictionary: dict
        key: none

    Returns:
        dict
    """
    return dictionary


def get_dictkey_auto_fn(key: Optional[Union[str, List[str]]]) -> Callable:
    """
    Function generator for sub-dict preparation from dict
        based on predefined keys.
    Args:
        key: keys

    Returns:
        function
    """
    if isinstance(key, str):
        return get_key_str
    elif isinstance(key, (list, tuple)):
        return get_key_list
    elif isinstance(key, dict):
        return get_key_dict
    else:
        return get_key_none
