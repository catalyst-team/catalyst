"""
IoU metric. Jaccard metric refers to IoU here, same functionality.
"""

from typing import List
from functools import partial

import torch

from catalyst.utils.torch import get_activation_fn


def iou(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    # values are discarded, only None check
    # used for compatibility with BatchMetricCallback
    classes: List[str] = None,
    eps: float = 1e-7,
    threshold: float = None,
    activation: str = "Sigmoid",
) -> torch.Tensor:
    """
    Args:
        outputs (torch.Tensor): A list of predicted elements
        targets (torch.Tensor):  A list of elements that are to be predicted
        classes (List[str]): if classes are specified
            we reduce across all dims except channels
        eps (float): epsilon to avoid zero division
        threshold (float): threshold for outputs binarization
        activation (str): An torch.nn activation applied to the outputs.
            Must be one of ["none", "Sigmoid", "Softmax2d"]

    Returns:
        Union[float, List[float]]: IoU (Jaccard) score(s)
    """
    activation_fn = get_activation_fn(activation)
    outputs = activation_fn(outputs)

    if threshold is not None:
        outputs = (outputs > threshold).float()

    # TODO: fix classes issue
    # ! fix backward compatibility
    if classes is not None:
        # if classes are specified we reduce across all dims except channels
        sum_strategy = partial(torch.sum, dim=[0, 2, 3])
    else:
        sum_strategy = torch.sum

    intersection = sum_strategy(targets * outputs)
    union = sum_strategy(targets) + sum_strategy(outputs)
    # this looks a bit awkward but `eps * (union == 0)` term
    # makes sure that if I and U are both 0, than IoU == 1
    # and if U != 0 and I == 0 the eps term in numerator is zeroed out
    # i.e. (0 + eps) / (U - 0 + eps) doesn't happen
    output_iou = (intersection + eps * (union == 0)) / (
        union - intersection + eps
    )

    return output_iou


jaccard = iou

__all__ = ["iou", "jaccard"]
