from typing import Optional, Tuple
from functools import partial

import torch


def _get_segmentation_stats(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    class_dim: int = 1,
    threshold: float = None,
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

    Returns: segmentation stats

    Examples:
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
        >>> _get_segmentation_stats(
        >>>                         outputs=pred,
        >>>                         targets=targets,
        >>>                         class_dim=1,
        >>>                         threshold=0.5,
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


def iou(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    class_dim: int = 1,
    threshold: float = None,
    eps: float = 1e-7,
) -> torch.Tensor:
    """
    Computes the iou/jaccard score,
    iou score = intersection / union = tp / (tp + fp + fn)

    Args:
        outputs: [N; K; ...] tensor that for each of the N examples
            indicates the probability of the example belonging to each of
            the K classes, according to the model.
        targets:  binary [N; K; ...] tensor that encodes which of the K
            classes are associated with the N-th input
        class_dim: indicates class dimention (K) for
            ``outputs`` and ``targets`` tensors (default = 1)
        threshold: threshold for outputs binarization
        eps: epsilon to avoid zero division

    Returns:
        IoU (Jaccard) score for each class

    Examples:
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
        >>> )
        tensor([0.0000, 0.0000, 1.0000, 1.0000, 1.0000, 0.5])
    """
    tp, fp, fn = _get_segmentation_stats(
        outputs=outputs,
        targets=targets,
        class_dim=class_dim,
        threshold=threshold,
    )
    union = tp + fp + fn
    score = (tp + eps * (union == 0).float()) / (tp + fp + fn + eps)
    return score


def dice(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    class_dim: int = 1,
    threshold: float = None,
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
            ``outputs`` and ``targets`` tensors (default = 1)
        threshold: threshold for outputs binarization
        eps: epsilon to avoid zero division

    Returns:
        Dice score for each class

    Examples:
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
        >>> )
        tensor([0.0000, 0.0000, 1.0000, 1.0000, 1.0000, 0.6667])
    """
    tp, fp, fn = _get_segmentation_stats(
        outputs=outputs,
        targets=targets,
        class_dim=class_dim,
        threshold=threshold,
    )
    union = tp + fp + fn
    score = (2 * tp + eps * (union == 0).float()) / (2 * tp + fp + fn + eps)
    return score


def trevsky(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float,
    beta: Optional[float] = None,
    class_dim: int = 1,
    threshold: float = None,
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
        eps: epsilon to avoid zero division

    Returns:
        Trevsky score for each class

    Examples:
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
        >>>         outputs=pred,
        >>>         targets=targets,
        >>>         alpha=0.2,
        >>>         class_dim=1,
        >>>         threshold=0.5,
        >>> )
        tensor([0.0000, 0.0000, 1.0000, 1.0000, 1.0000, 0.8333])
    """
    assert 0 < alpha < 1  # I am not sure about this
    if beta is None:
        beta = 1 - alpha
    tp, fp, fn = _get_segmentation_stats(
        outputs=outputs,
        targets=targets,
        class_dim=class_dim,
        threshold=threshold,
    )
    union = tp + fp + fn
    score = (tp + eps * (union == 0).float()) / (
        tp + fp * beta + fn * alpha + eps
    )
    return score
