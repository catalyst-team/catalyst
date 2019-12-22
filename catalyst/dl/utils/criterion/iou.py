from typing import List, Union  # isort:skip

import torch

from catalyst.utils import get_activation_fn


def iou(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    # values are discarded, only None check
    # used for compatibility with MultiMetricCallback
    classes: List[str] = None,
    eps: float = 1e-7,
    threshold: float = None,
    activation: str = "Sigmoid"
) -> Union[float, List[float]]:
    """
    Args:
        outputs (torch.Tensor): A list of predicted elements
        targets (torch.Tensor):  A list of elements that are to be predicted
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

    # if classes are specified we reduce across all dims except channels
    red_dim = list(range(len(targets.shape))) if classes is None else [0, 2, 3]

    intersection = torch.sum(targets * outputs, red_dim)
    union = torch.sum(targets, red_dim) + torch.sum(outputs, red_dim)
    # this looks a bit awkward but `eps * (union == 0)` term
    # makes sure that if I and U are both 0, than IoU == 1
    # and if U != 0 and I == 0 the eps term in numerator is zeroed out
    # i.e. (0 + eps) / (U - 0 + eps) doesn't happen
    iou = (intersection + eps * (union == 0)) / (union - intersection + eps)

    return iou


jaccard = iou

__all__ = ["iou", "jaccard"]
