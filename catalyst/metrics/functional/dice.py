from functools import partial

import numpy as np
import torch


def dice(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    class_dim: int = 1,
    threshold: float = None,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Computes the dice score.

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
        Dice score

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
        >>>     outputs=pred,
        >>>     targets=targets,
        >>>     class_dim=1,
        >>>     threshold=0.5,
        >>> )
        tensor([0.0000, 0.0000, 1.0000, 1.0000, 1.0000, 0.66666])
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
    # makes sure that if I and U are both 0, than Dice == 1
    # and if U != 0 and I == 0 the eps term in numerator is zeroed out
    # i.e. (0 + eps) / (U - 0 + eps) doesn't happen
    dice_score = (2 * intersection + eps * (union == 0).float()) / (union + eps)

    return dice_score


# @TODO: remove
def calculate_dice(
    true_positives: np.array, false_positives: np.array, false_negatives: np.array,
) -> np.array:
    """
    Calculate list of Dice coefficients.

    Args:
        true_positives: true positives numpy tensor
        false_positives: false positives numpy tensor
        false_negatives: false negatives numpy tensor

    Returns:
        np.array: dice score

    Raises:
        ValueError: if `dice` is out of [0; 1] bounds
    """
    epsilon = 1e-7

    dice_metric = (2 * true_positives + epsilon) / (
        2 * true_positives + false_positives + false_negatives + epsilon
    )

    if not np.all(dice_metric <= 1):
        raise ValueError("Dice index should be less or equal to 1")

    if not np.all(dice_metric > 0):
        raise ValueError("Dice index should be more than 1")

    return dice_metric


__all__ = ["dice", "calculate_dice"]
