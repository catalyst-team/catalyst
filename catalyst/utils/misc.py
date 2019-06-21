from typing import Iterable, Any

import copy
import collections
import numpy as np
from itertools import tee


def pairwise(iterable: Iterable[Any]) -> Iterable[Any]:
    """
    Iterate sequences by pairs

    Args:
        iterable: Any iterable sequence

    Returns:
        pairwise iterator

    Examples:
        >>> for i in pairwise([1, 2, 5, -3]):
        >>>     print(i)
        (1, 2)
        (2, 5)
        (5, -3)
    """
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def make_tuple(tuple_like):
    tuple_like = (
        tuple_like if isinstance(tuple_like, (list, tuple)) else
        (tuple_like, tuple_like)
    )
    return tuple_like


def merge_dicts(*dicts: dict) -> dict:
    """
    Recursive dict merge.
    Instead of updating only top-level keys,
    ``merge_dicts`` recurses down into dicts nested
    to an arbitrary depth, updating keys.

    Args:
        *dicts: several dictionaries to merge

    Returns:
        dict: deep-merged dictionary
    """
    assert len(dicts) > 1

    dict_ = copy.deepcopy(dicts[0])

    for merge_dict in dicts[1:]:
        merge_dict = merge_dict or {}
        for k, v in merge_dict.items():
            if (
                k in dict_ and isinstance(dict_[k], dict)
                and isinstance(merge_dict[k], collections.Mapping)
            ):
                dict_[k] = merge_dicts(dict_[k], merge_dict[k])
            else:
                dict_[k] = merge_dict[k]

    return dict_


def append_dict(dict1, dict2):
    """
    Appends dict2 with the same keys as dict1 to dict1
    """
    for key in dict1.keys():
        dict1[key] = np.concatenate((dict1[key], dict2[key]))
    return dict1
