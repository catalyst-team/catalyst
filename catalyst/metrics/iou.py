from functools import partial

import torch


def iou(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    class_dim: int = 1,
    threshold: float = None,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Computes the iou/jaccard score.

    Args:
        outputs: [N; K; ...] tensor that for each of the N examples
            indicates the probability of the example belonging to each of
            the K classes, according to the model.
        targets:  binary [N; K; ...] tensort that encodes which of the K
            classes are associated with the N-th input
        class_dim: indicates class dimention (K) for
            ``outputs`` and ``targets`` tensors (default = 1)
        threshold: threshold for outputs binarization
        eps: epsilon to avoid zero division

    Returns:
        IoU (Jaccard) score

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
    if threshold is not None:
        outputs = (outputs > threshold).float()

    num_dims = len(outputs.shape)
    assert num_dims > 2, "shape mismatch, please check the docs for more info"
    assert outputs.shape == targets.shape, "shape mismatch, please check the docs for more info"
    dims = list(range(num_dims))
    # support negative index
    if class_dim < 0:
        class_dim = num_dims + class_dim
    dims.pop(class_dim)
    sum_fn = partial(torch.sum, dim=dims)

    intersection = sum_fn(targets * outputs)
    union = sum_fn(targets) + sum_fn(outputs)
    # this looks a bit awkward but `eps * (union == 0)` term
    # makes sure that if I and U are both 0, than IoU == 1
    # and if U != 0 and I == 0 the eps term in numerator is zeroed out
    # i.e. (0 + eps) / (U - 0 + eps) doesn't happen
    iou_score = (intersection + eps * (union == 0).float()) / (union - intersection + eps)

    return iou_score


jaccard = iou

__all__ = ["iou", "jaccard"]
