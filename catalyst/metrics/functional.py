from typing import Tuple

import torch


def get_binary_statistics(
    predictions: torch.Tensor, targets: torch.Tensor
) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    """
    Computes the number of true positive, false positive, true negative,
    false negative and support for a binary classification problem.

    Args:
        predictions (torch.Tensor): model predictions
        targets (torch.Tensor):

    Returns:

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
