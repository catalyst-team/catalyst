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


class Seeder:
    """
    A random seed generator.

    Given an initial seed,
    the seeder can be called continuously to sample a single
    or a batch of random seeds.

    .. note::

        The seeder creates an independent RandomState to generate random
        numbers. It does not affect the RandomState in ``np.random``.

    Example::
        >>> seeder = Seeder(init_seed=0)
        >>> seeder(size=5)
        [209652396, 398764591, 924231285, 1478610112, 441365315]

    """

    def __init__(self, init_seed: int = 0, max_seed: int = None):
        """
        Initialize the seeder.

        Args:
            init_seed (int, optional):
                Initial seed for generating random seeds. Default: ``0``.
        """
        assert isinstance(init_seed, int) and init_seed >= 0, \
            f"expected non-negative integer, got {init_seed}"

        self.rng = np.random.RandomState(seed=init_seed)

        # Upper bound for sampling new random seeds
        self.max = max_seed or np.iinfo(np.int32).max

    def __call__(self, size=1):
        """
        Return the sampled random seeds according to the given size.

        Args:
            size (int or list): The size of random seeds to sample.

        Returns
        -------
        seeds : list
            a list of sampled random seeds.
        """
        seeds = self.rng.randint(low=0, high=self.max, size=size).tolist()

        return seeds
