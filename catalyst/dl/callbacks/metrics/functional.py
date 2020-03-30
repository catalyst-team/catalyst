import numpy as np


def calculate_dice(
    true_positives: np.array,
    false_positives: np.array,
    false_negatives: np.array,
) -> np.array:
    """Calculate list of Dice coefficients.

    Args:
        true_positives:
        false_positives:
        false_negatives:

    Returns:
        np.array: dice score
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
