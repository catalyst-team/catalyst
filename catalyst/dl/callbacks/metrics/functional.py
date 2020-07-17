from typing import List

import numpy as np


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

    dice = (2 * true_positives + epsilon) / (
        2 * true_positives + false_positives + false_negatives + epsilon
    )

    if not np.all(dice <= 1):
        raise ValueError("Dice index should be less or equal to 1")

    if not np.all(dice > 0):
        raise ValueError("Dice index should be more than 1")

    return dice


def get_default_topk_args(num_classes: int) -> List[int]:
    """Calculate list params for ``Accuracy@k`` and ``mAP@k``.

    Examples:
        >>> get_default_topk_args(num_classes=4)
        >>> [1, 3]
        >>> get_default_topk_args(num_classes=8)
        >>> [1, 3, 5]

    Args:
        num_classes (int): number of classes

    Returns:
        iterable: array of accuracy arguments
    """
    result = [1]

    if num_classes is None:
        return result

    if num_classes > 3:
        result.append(3)
    if num_classes > 5:
        result.append(5)

    return result
