from typing import Tuple

import torch
from torch import Tensor


def get_binary_statistics(
    predictions: Tensor, targets: Tensor
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """
    Computes the number of true positive, false positive, true negative,
    false negative and support for a binary classification problem.

    Args:
        predictions (Tensor): Estimated targets as predicted by a model.
        targets (Tensor): Ground truth (correct) target values.

    Returns:
        Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]: stats
    """
    true_positive = ((predictions == 1) * (targets == 1)).to(torch.long).sum()
    false_positive = ((predictions == 1) * (targets != 1)).to(torch.long).sum()
    true_negative = ((predictions != 1) * (targets != 1)).to(torch.long).sum()
    false_negative = ((predictions != 1) * (targets == 1)).to(torch.long).sum()
    support = (targets == 1).to(torch.long).sum()

    return (
        true_positive,
        false_positive,
        true_negative,
        false_negative,
        support,
    )
