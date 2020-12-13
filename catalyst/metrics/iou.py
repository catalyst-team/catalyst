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
        >>> iou(
        >>>     outputs=torch.tensor([
        >>>         [1, 0, 0],
        >>>         [0, 1, 0],
        >>>         [0, 0, 1],
        >>>     ]),
        >>>     targets=torch.tensor([
        >>>         [1, 0, 0],
        >>>         [0, 1, 0],
        >>>         [0, 0, 1],
        >>>     ]),
        >>>     threshold=0.5,
        >>> )
        tensor(1.0)
        >>> iou(
        >>>     outputs=torch.tensor([
        >>>         [1, 0, 0],
        >>>         [0, 1, 0],
        >>>         [0, 0, 1],
        >>>     ]),
        >>>     targets=torch.tensor([
        >>>         [1, 0, 0],
        >>>         [0, 1, 0],
        >>>         [0, 0, 0],
        >>>     ]),
        >>>     threshold=0.5,
        >>> )
        tensor(0.6667)
    """
    if threshold is not None:
        outputs = (outputs > threshold).float()

    num_dims = len(outputs.shape)
    assert num_dims > 2, "shape mismatch, please check the docs for more info"
    assert (
        outputs.shape == targets.shape
    ), "shape mismatch, please check the docs for more info"
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
    iou_score = (intersection + eps * (union == 0).float()) / (
        union - intersection + eps
    )

    return iou_score


jaccard = iou

__all__ = ["iou", "jaccard"]
