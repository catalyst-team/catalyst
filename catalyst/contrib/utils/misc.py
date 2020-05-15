from typing import Any, Iterable, Optional, List
from itertools import tee


def pairwise(iterable: Iterable[Any]) -> Iterable[Any]:
    """Iterate sequences by pairs.

    Examples:
        >>> for i in pairwise([1, 2, 5, -3]):
        >>>     print(i)
        (1, 2)
        (2, 5)
        (5, -3)

    Args:
        iterable: Any iterable sequence

    Returns:
        pairwise iterator
    """
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def make_tuple(tuple_like):
    """Creates a tuple if given ``tuple_like`` value isn't list or tuple.

    Returns:
        tuple or list
    """
    tuple_like = (
        tuple_like
        if isinstance(tuple_like, (list, tuple))
        else (tuple_like, tuple_like)
    )
    return tuple_like


def args_are_not_none(*args: Optional[Any]) -> bool:
    """Check that all arguments are not ``None``.

    Args:
        *args (Any): values

    Returns:
         bool: True if all value were not None, False otherwise
    """
    if args is None:
        return False

    for arg in args:
        if arg is None:
            return False

    return True


def find_all(x: Iterable[Any], x0: Any) -> List[int]:
    """

    Args:
        x: list of any
        x0: query element

    Returns: indices of the all elements equal x0

    """
    inds = [i for i, el in enumerate(x) if el == x0]
    return inds


__all__ = ["args_are_not_none", "make_tuple", "pairwise", "find_all"]
