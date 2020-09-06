from typing import List, Union

from torch import int as tint, long, short, Tensor


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
