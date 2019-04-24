from typing import Iterable, Any, Optional

import copy
import random
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
        for k, v in merge_dict.items():
            if (
                k in dict_ and isinstance(dict_[k], dict)
                and isinstance(merge_dict[k], collections.Mapping)
            ):
                dict_[k] = merge_dicts(dict_[k], merge_dict[k])
            else:
                dict_[k] = merge_dict[k]

    return dict_


def set_global_seed(seed: int) -> None:
    """
    Sets random seed into PyTorch, TensorFlow, Numpy and Random

    Args:
        seed: random seed
    """
    try:
        import torch
    except ImportError:
        pass
    else:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    try:
        import tensorflow as tf
    except ImportError:
        pass
    else:
        tf.set_random_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def boolean_flag(
    parser, name: str, default: bool = False, help: str = None
) -> None:
    """
    Add a boolean flag to argparse parser.

    Args:
        parser (argparse.Parser): parser to add the flag to
        name (str): --<name> will enable the flag,
            while --no-<name> will disable it
        default (bool, optional): default value of the flag
        help (str): help string for the flag
    """
    dest = name.replace("-", "_")
    parser.add_argument(
        "--" + name,
        action="store_true",
        default=default,
        dest=dest,
        help=help
    )
    parser.add_argument("--no-" + name, action="store_false", dest=dest)


class FrozenClass(object):
    """
    Class which prohibit ``__setattr__`` on existing attributes

    Examples:
        >>> class RunnerState(FrozenClass):
    """
    __isfrozen = False

    def __setattr__(self, key, value):
        if self.__isfrozen and not hasattr(self, key):
            raise TypeError("%r is a frozen class" % self)
        object.__setattr__(self, key, value)

    def _freeze(self):
        self.__isfrozen = True


def args_are_not_none(*args: Optional[Any]) -> bool:
    """
    Check that all arguments are not None
    Args:
        *args (Any): values
    Returns:
         bool: True if all value were not None, False otherwise
    """
    result = args is not None
    if not result:
        return result

    for arg in args:
        if arg is None:
            result = False
            break

    return result
