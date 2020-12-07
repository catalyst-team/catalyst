import numpy as np

import torch


# @TODO: make it work in "per class" mode
def dice(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    eps: float = 1e-7,
    threshold: float = None,
) -> float:
    """Computes the dice score.

    Args:
        outputs:  a list of predicted elements
            with shape [batch_size; num_classes; ...]
        targets: a list of elements that are to be predicted
            with shape [batch_size; num_classes; ...]
        eps: epsilon
        threshold: threshold for outputs binarization

    Returns:
        Dice score
    """

    if threshold is not None:
        outputs = (outputs > threshold).float()

    intersection = torch.sum(targets * outputs)
    union = torch.sum(targets) + torch.sum(outputs)
    # this looks a bit awkward but `eps * (union == 0)` term
    # makes sure that if I and U are both 0, than Dice == 1
    # and if U != 0 and I == 0 the eps term in numerator is zeroed out
    # i.e. (0 + eps) / (U - 0 + eps) doesn't happen
    dice_score = (2 * intersection + eps * (union == 0)) / (union + eps)

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
