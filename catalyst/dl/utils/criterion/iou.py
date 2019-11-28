import torch

from typing import List

from catalyst.utils import get_activation_fn


def iou(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    classes: List[str] = None,
    eps: float = 1e-7,
    threshold: float = None,
    activation: str = "Sigmoid"
):
    """
    Args:
        outputs (torch.Tensor): A list of predicted elements
        targets (torch.Tensor):  A list of elements that are to be predicted
        eps (float): epsilon to avoid zero division
        threshold (float): threshold for outputs binarization
        activation (str): An torch.nn activation applied to the outputs.
            Must be one of ["none", "Sigmoid", "Softmax2d"]

    Returns:
        float or List[float]: IoU (Jaccard) score(s)
    """
    activation_fn = get_activation_fn(activation)
    outputs = activation_fn(outputs)

    if threshold is not None:
        outputs = (outputs > threshold).float()
        
    reduction_channels = (0, 1, 2, 3) if classes is None else (0, 2, 3)
    intersection = torch.sum(targets * outputs, reduction_channels)
    union = torch.sum(targets, reduction_channels) + torch.sum(outputs, reduction_channels)
    iou = (intersection + eps) / (union - intersection + eps)

    return iou


jaccard = iou

__all__ = ["iou", "jaccard"]
