from typing import Callable, List, Optional, Tuple
from functools import partial

import torch


def get_segmentation_statistics(
    outputs: torch.Tensor, targets: torch.Tensor, class_dim: int = 1, threshold: float = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes true positive, false positive, false negative
    for a multilabel segmentation problem.

    Args:
        outputs: [N; K; ...] tensor that for each of the N examples
            indicates the probability of the example belonging to each of
            the K classes, according to the model.
        targets:  binary [N; K; ...] tensor that encodes which of the K
            classes are associated with the N-th input
        class_dim: indicates class dimention (K) for
            ``outputs`` and ``targets`` tensors (default = 1)
        threshold: threshold for outputs binarization

    Returns:
        Segmentation stats

    Example:
        >>> size = 4
        >>> half_size = size // 2
        >>> shape = (1, 1, size, size)
        >>> empty = torch.zeros(shape)
        >>> full = torch.ones(shape)
        >>> left = torch.ones(shape)
        >>> left[:, :, :, half_size:] = 0
        >>> right = torch.ones(shape)
        >>> right[:, :, :, :half_size] = 0
        >>> top_left = torch.zeros(shape)
        >>> top_left[:, :, :half_size, :half_size] = 1
        >>> pred = torch.cat([empty, left, empty, full, left, top_left], dim=1)
        >>> targets = torch.cat([full, right, empty, full, left, left], dim=1)
        >>> get_segmentation_statistics(
        >>>     outputs=pred,
        >>>     targets=targets,
        >>>     class_dim=1,
        >>>     threshold=0.5,
        >>> )
        (tensor([ 0.,  0.,  0., 16.,  8.,  4.]),
        tensor([0., 8., 0., 0., 0., 0.]),
        tensor([16.,  8.,  0.,  0.,  0.,  4.]))
    """
    assert outputs.shape == targets.shape, (
        f"targets(shape {targets.shape})"
        f" and outputs(shape {outputs.shape}) must have the same shape"
    )
    if threshold is not None:
        outputs = (outputs > threshold).float()

    n_dims = len(outputs.shape)
    dims = list(range(n_dims))
    # support negative index
    if class_dim < 0:
        class_dim = n_dims + class_dim
    dims.pop(class_dim)

    sum_per_class = partial(torch.sum, dim=dims)

    tp = sum_per_class(outputs * targets)
    class_union = sum_per_class(outputs) + sum_per_class(targets)
    class_union -= tp
    fp = sum_per_class(outputs * (1 - targets))
    fn = sum_per_class(targets * (1 - outputs))
    return tp, fp, fn


def _get_region_based_metrics(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    metric_fn: Callable,
    class_dim=None,
    threshold: float = None,
    mode: str = "per-class",
    weights: Optional[List[float]] = None,
) -> torch.Tensor:
    """
    Get aggregated metric

    Args:
        outputs: [N; K; ...] tensor that for each of the N examples
            indicates the probability of the example belonging to each of
            the K classes, according to the model.
        targets:  binary [N; K; ...] tensor that encodes which of the K
            classes are associated with the N-th input
        metric_fn: metric function, that get statistics and return score
        class_dim: indicates class dimention (K) for
            ``outputs`` and ``targets`` tensors (default = 1), if
            mode = "micro" means nothing
        threshold: threshold for outputs binarization
        mode: class summation strategy. Must be one of ['micro', 'macro',
            'weighted', 'per-class']. If mode='micro', classes are ignored,
             and metric are calculated generally. If mode='macro', metric are
             calculated per-class and than are averaged over all classes. If
             mode='weighted', metric are calculated per-class and than summed
             over all classes with weights. If mode='per-class', metric are
             calculated separately for all classes
        weights: class weights(for mode="weighted")

    Returns:
        computed metric
    """
    assert mode in ["per-class", "micro", "macro", "weighted"]
    segmentation_stats = get_segmentation_statistics(
        outputs=outputs, targets=targets, class_dim=class_dim, threshold=threshold,
    )
    if mode == "micro":
        segmentation_stats = [torch.sum(stats) for stats in segmentation_stats]
        metric = metric_fn(*segmentation_stats)

    metrics_per_class = metric_fn(*segmentation_stats)

    if mode == "macro":
        metric = torch.mean(metrics_per_class)
    elif mode == "weighted":
        assert len(weights) == len(segmentation_stats[0])
        device = metrics_per_class.device
        metrics = torch.tensor(weights).to(device) * metrics_per_class
        metric = torch.sum(metrics)
    elif mode == "per-class":
        metric = metrics_per_class

    return metric


def _iou(tp: torch.Tensor, fp: torch.Tensor, fn: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    union = tp + fp + fn
    score = (tp + eps * (union == 0).float()) / (tp + fp + fn + eps)
    return score


def _dice(tp: torch.Tensor, fp: torch.Tensor, fn: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    union = tp + fp + fn
    score = (2 * tp + eps * (union == 0).float()) / (2 * tp + fp + fn + eps)
    return score


def _trevsky(
    tp: torch.Tensor,
    fp: torch.Tensor,
    fn: torch.Tensor,
    alpha: float,
    beta: float,
    eps: float = 1e-7,
) -> torch.Tensor:
    union = tp + fp + fn
    score = (tp + eps * (union == 0).float()) / (tp + fp * beta + fn * alpha + eps)
    return score


def iou(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    class_dim: int = 1,
    threshold: float = None,
    mode: str = "per-class",
    weights: Optional[List[float]] = None,
    eps: float = 1e-7,
) -> torch.Tensor:
    """
    Computes the iou/jaccard score, iou score = intersection / union = tp / (tp + fp + fn)

    Args:
        outputs: [N; K; ...] tensor that for each of the N examples
            indicates the probability of the example belonging to each of
            the K classes, according to the model.
        targets:  binary [N; K; ...] tensor that encodes which of the K
            classes are associated with the N-th input
        class_dim: indicates class dimention (K) for
            ``outputs`` and ``targets`` tensors (default = 1), if
            mode = "micro" means nothing
        threshold: threshold for outputs binarization
        mode: class summation strategy. Must be one of ['micro', 'macro',
            'weighted', 'per-class']. If mode='micro', classes are ignored,
            and metric are calculated generally. If mode='macro', metric are
            calculated per-class and than are averaged over all classes. If
            mode='weighted', metric are calculated per-class and than summed
            over all classes with weights. If mode='per-class', metric are
            calculated separately for all classes
        weights: class weights(for mode="weighted")
        eps: epsilon to avoid zero division

    Returns:
        IoU (Jaccard) score for each class(if mode='weighted') or aggregated IOU

    Example:
        >>> size = 4
        >>> half_size = size // 2
        >>> shape = (1, 1, size, size)
        >>> empty = torch.zeros(shape)
        >>> full = torch.ones(shape)
        >>> left = torch.ones(shape)
        >>> left[:, :, :, half_size:] = 0
        >>> right = torch.ones(shape)
        >>> right[:, :, :, :half_size] = 0
        >>> top_left = torch.zeros(shape)
        >>> top_left[:, :, :half_size, :half_size] = 1
        >>> pred = torch.cat([empty, left, empty, full, left, top_left], dim=1)
        >>> targets = torch.cat([full, right, empty, full, left, left], dim=1)
        >>> iou(
        >>>     outputs=pred,
        >>>     targets=targets,
        >>>     class_dim=1,
        >>>     threshold=0.5,
        >>>     mode="per-class"
        >>> )
        tensor([0.0000, 0.0000, 1.0000, 1.0000, 1.0000, 0.5])
    """
    metric_fn = partial(_iou, eps=eps)
    score = _get_region_based_metrics(
        outputs=outputs,
        targets=targets,
        metric_fn=metric_fn,
        class_dim=class_dim,
        threshold=threshold,
        mode=mode,
        weights=weights,
    )
    return score


def dice(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    class_dim: int = 1,
    threshold: float = None,
    mode: str = "per-class",
    weights: Optional[List[float]] = None,
    eps: float = 1e-7,
) -> torch.Tensor:
    """
    Computes the dice score,
    dice score = 2 * intersection / (intersection + union)) = \
    = 2 * tp / (2 * tp + fp + fn)

    Args:
        outputs: [N; K; ...] tensor that for each of the N examples
            indicates the probability of the example belonging to each of
            the K classes, according to the model.
        targets:  binary [N; K; ...] tensor that encodes which of the K
            classes are associated with the N-th input
        class_dim: indicates class dimention (K) for
            ``outputs`` and ``targets`` tensors (default = 1), if
            mode = "micro" means nothing
        threshold: threshold for outputs binarization
        mode: class summation strategy. Must be one of ['micro', 'macro',
            'weighted', 'per-class']. If mode='micro', classes are ignored,
            and metric are calculated generally. If mode='macro', metric are
            calculated per-class and than are averaged over all classes. If
            mode='weighted', metric are calculated per-class and than summed
            over all classes with weights. If mode='per-class', metric are
            calculated separately for all classes
        weights: class weights(for mode="weighted")
        eps: epsilon to avoid zero division

    Returns:
        Dice score for each class(if mode='weighted') or aggregated Dice

    Example:
        >>> size = 4
        >>> half_size = size // 2
        >>> shape = (1, 1, size, size)
        >>> empty = torch.zeros(shape)
        >>> full = torch.ones(shape)
        >>> left = torch.ones(shape)
        >>> left[:, :, :, half_size:] = 0
        >>> right = torch.ones(shape)
        >>> right[:, :, :, :half_size] = 0
        >>> top_left = torch.zeros(shape)
        >>> top_left[:, :, :half_size, :half_size] = 1
        >>> pred = torch.cat([empty, left, empty, full, left, top_left], dim=1)
        >>> targets = torch.cat([full, right, empty, full, left, left], dim=1)
        >>> dice(
        >>>      outputs=pred,
        >>>      targets=targets,
        >>>      class_dim=1,
        >>>      threshold=0.5,
        >>>      mode="per-class"
        >>> )
        tensor([0.0000, 0.0000, 1.0000, 1.0000, 1.0000, 0.6667])
    """
    metric_fn = partial(_dice, eps=eps)
    score = _get_region_based_metrics(
        outputs=outputs,
        targets=targets,
        metric_fn=metric_fn,
        class_dim=class_dim,
        threshold=threshold,
        mode=mode,
        weights=weights,
    )
    return score


def trevsky(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float,
    beta: Optional[float] = None,
    class_dim: int = 1,
    threshold: float = None,
    mode: str = "per-class",
    weights: Optional[List[float]] = None,
    eps: float = 1e-7,
) -> torch.Tensor:
    """
    Computes the trevsky score,
    trevsky score = tp / (tp + fp * beta + fn * alpha)

    Args:
        outputs: [N; K; ...] tensor that for each of the N examples
            indicates the probability of the example belonging to each of
            the K classes, according to the model.
        targets:  binary [N; K; ...] tensor that encodes which of the K
            classes are associated with the N-th input
        alpha: false negative coefficient, bigger alpha bigger penalty for
            false negative. Must be in (0, 1)
        beta: false positive coefficient, bigger alpha bigger penalty for false
            positive. Must be in (0, 1), if None beta = (1 - alpha)
        class_dim: indicates class dimention (K) for
            ``outputs`` and ``targets`` tensors (default = 1)
        threshold: threshold for outputs binarization
        mode: class summation strategy. Must be one of ['micro', 'macro',
            'weighted', 'per-class']. If mode='micro', classes are ignored,
            and metric are calculated generally. If mode='macro', metric are
            calculated per-class and than are averaged over all classes. If
            mode='weighted', metric are calculated per-class and than summed
            over all classes with weights. If mode='per-class', metric are
            calculated separately for all classes
        weights: class weights(for mode="weighted")
        eps: epsilon to avoid zero division

    Returns:
        Trevsky score for each class(if mode='weighted') or aggregated score

    Example:
        >>> size = 4
        >>> half_size = size // 2
        >>> shape = (1, 1, size, size)
        >>> empty = torch.zeros(shape)
        >>> full = torch.ones(shape)
        >>> left = torch.ones(shape)
        >>> left[:, :, :, half_size:] = 0
        >>> right = torch.ones(shape)
        >>> right[:, :, :, :half_size] = 0
        >>> top_left = torch.zeros(shape)
        >>> top_left[:, :, :half_size, :half_size] = 1
        >>> pred = torch.cat([empty, left, empty, full, left, top_left], dim=1)
        >>> targets = torch.cat([full, right, empty, full, left, left], dim=1)
        >>> trevsky(
        >>>     outputs=pred,
        >>>     targets=targets,
        >>>     alpha=0.2,
        >>>     class_dim=1,
        >>>     threshold=0.5,
        >>>     mode="per-class"
        >>> )
        tensor([0.0000, 0.0000, 1.0000, 1.0000, 1.0000, 0.8333])
    """
    # assert 0 < alpha < 1  # I am not sure about this
    if beta is None:
        assert 0 < alpha < 1, "if beta=None, alpha must be in (0, 1)"
        beta = 1 - alpha
    metric_fn = partial(_trevsky, alpha=alpha, beta=beta, eps=eps)
    score = _get_region_based_metrics(
        outputs=outputs,
        targets=targets,
        metric_fn=metric_fn,
        class_dim=class_dim,
        threshold=threshold,
        mode=mode,
        weights=weights,
    )
    return score


__all__ = ["iou", "dice", "trevsky", "get_segmentation_statistics"]
