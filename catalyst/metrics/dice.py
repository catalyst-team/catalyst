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
        >>> dice(
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
        >>> dice(
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
        tensor(0.8000)
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
    # makes sure that if I and U are both 0, than Dice == 1
    # and if U != 0 and I == 0 the eps term in numerator is zeroed out
    # i.e. (0 + eps) / (U - 0 + eps) doesn't happen
    dice_score = (2 * intersection + eps * (union == 0).float()) / (
        union + eps
    )

    return dice_score


# @TODO: remove
def calculate_dice(
    true_positives: np.array,
    false_positives: np.array,
    false_negatives: np.array,
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
