from typing import Callable, List, Optional, Union
from functools import partial

import torch

from catalyst.utils.torch import get_activation_fn


def _get_segmentation_stats(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    class_dim: Optional[int] = None,
    threshold: float = None,
    activation: str = "Sigmoid",
) -> List[torch.Tensor]:
    assert outputs.shape == targets.shape, (
        f"targets(shape {targets.shape})"
        f" and outputs(shape {outputs.shape})\
        must have the same shape"
    )

    activation_fn = get_activation_fn(activation)
    outputs = activation_fn(outputs)

    if threshold is not None:
        outputs = (outputs > threshold).float()

    if class_dim is None:
        outputs = outputs.unsqueeze(0)
        targets = targets.unsqueeze(0)
        class_dim = 0
    n_dims = len(outputs.shape)
    dims = list(range(n_dims))
    # support negative index
    if class_dim < 0:
        class_dim = n_dims + class_dim
    dims.pop(class_dim)

    sum_per_class = partial(torch.sum, dim=dims)

    class_intersection = sum_per_class(outputs * targets)
    class_union = sum_per_class(outputs) + sum_per_class(targets)
    class_union -= class_intersection
    class_fp = sum_per_class(outputs * (1 - targets))
    class_fn = sum_per_class(targets * (1 - outputs))
    return class_intersection, class_union, class_fp, class_fn


def _get_region_metrics(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    metric_function: Callable,
    class_dim=None,
    threshold: float = None,
    activation: str = "Sigmoid",
    mode: Union[str, List[float]] = "macro",
    weights: Optional[List[float]] = None,
) -> torch.Tensor:
    """
    Calculate region based metric

    Args:
        outputs: predicted elements
        targets: elements that are to be predicted
        metric_function: function that get segmentation statistics and
        calculate metric
        class_dim: class dim, if input is [batch_size, n_classes, H, W] you
        should make class_dim=1
        threshold: threshold for outputs binarization
        activation: An torch.nn activation applied to the outputs.Must be one
        of ["none", "Sigmoid", "Softmax2d"]
        mode: class summation strategy. Must be one of ["macro", "micro",
        "weighted"]
        weights: class weights(for mode="weighted")

    Returns: score
    """
    assert mode in ["macro", "micro", "weighted"]
    segmentation_stats = _get_segmentation_stats(
        outputs=outputs,
        targets=targets,
        class_dim=class_dim,
        threshold=threshold,
        activation=activation,
    )
    if mode == "macro":
        segmentation_stat = [torch.sum(stats) for stats in segmentation_stats]
        score = metric_function(segmentation_stat)
        return score

    n_classes = len(segmentation_stats[0])
    if mode == "micro":
        weights = [1.0 / n_classes] * n_classes
    else:
        assert len(weights) == n_classes
    score = 0
    for weight, class_idx in zip(weights, range(n_classes)):
        segmentation_stat = [stats[class_idx] for stats in segmentation_stats]
        score += metric_function(segmentation_stat) * weight
    return score


def _iou(
    segmentation_stats: List[torch.Tensor], eps: float = 1e-7
) -> torch.Tensor:
    intersection, union, _, _ = segmentation_stats
    iou_score = (intersection + eps * (union == 0)) / (union + eps)
    return iou_score


def _dice(segmentation_stats: List[torch.Tensor], eps: float = 1e-7):
    intersection, union, _, _ = segmentation_stats
    dice_score = (2 * intersection + eps * (union == 0)) / (union + eps)
    return dice_score


def _trevsky(
    segmentation_stats: List[torch.Tensor],
    alpha: float,
    beta: float,
    eps: float = 1e-7,
):
    intersection, _, fp, fn = segmentation_stats
    trevsky_score = intersection / (
        intersection + fn * alpha + beta * fp + eps
    )
    return trevsky_score


def iou(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    class_dim=None,
    threshold: float = None,
    activation: str = "Sigmoid",
    mode: Union[str, List[float]] = "macro",
    weights: Optional[List[float]] = None,
    eps: float = 1e-7,
):
    metric_function = partial(_iou, eps=eps)
    score = _get_region_metrics(
        outputs=outputs,
        targets=targets,
        metric_function=metric_function,
        class_dim=class_dim,
        threshold=threshold,
        activation=activation,
        mode=mode,
        weights=weights,
    )
    return score


def dice(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    class_dim=None,
    threshold: float = None,
    activation: str = "Sigmoid",
    mode: Union[str, List[float]] = "macro",
    weights: Optional[List[float]] = None,
    eps: float = 1e-7,
):
    metric_function = partial(_dice, eps=eps)
    score = _get_region_metrics(
        outputs=outputs,
        targets=targets,
        metric_function=metric_function,
        class_dim=class_dim,
        threshold=threshold,
        activation=activation,
        mode=mode,
        weights=weights,
    )
    return score


def trevsky(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float,
    beta: Optional[float] = None,
    class_dim=None,
    threshold: float = None,
    activation: str = "Sigmoid",
    mode: Union[str, List[float]] = "macro",
    weights: Optional[List[float]] = None,
    eps: float = 1e-7,
):
    assert 0 < alpha < 1
    if beta is None:
        beta = 1 - alpha
    metric_function = partial(_dice, alpha=alpha, beta=beta, eps=eps)
    score = _get_region_metrics(
        outputs=outputs,
        targets=targets,
        metric_function=metric_function,
        class_dim=class_dim,
        threshold=threshold,
        activation=activation,
        mode=mode,
        weights=weights,
    )
    return score
