from typing import Any, Iterable, List, Union

import numpy as np

from torch import int as tint, long, short, Tensor


def find_value_ids(it: Iterable[Any], value: Any) -> List[int]:
    """
    Args:
        it: list of any
        value: query element

    Returns:
        indices of the all elements equal x0
    """
    if isinstance(it, np.ndarray):
        inds = list(np.where(it == value)[0])
    else:  # could be very slow
        inds = [i for i, el in enumerate(it) if el == value]
    return inds


def convert_labels2list(labels: Union[Tensor, List[int]]) -> List[int]:
    """
    This function allows to work with 2 types of indexing:
    using a integer tensor and a list of indices.

    Args:
        labels: labels of batch samples

    Returns:
        labels of batch samples in the aligned format

    Raises:
        TypeError: if type of input labels is not tensor and list
    """
    if isinstance(labels, Tensor):
        labels = labels.squeeze()
        assert (len(labels.shape) == 1) and (
            labels.dtype in [short, tint, long]
        ), "Labels cannot be interpreted as indices."
        labels_list = labels.tolist()
    elif isinstance(labels, list):
        labels_list = labels.copy()
    else:
        raise TypeError(f"Unexpected type of labels: {type(labels)}).")

    return labels_list


__all__ = ["find_value_ids", "convert_labels2list"]
