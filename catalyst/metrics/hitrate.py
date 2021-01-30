"""
Hitrate metric:
    * :func:`hitrate`
"""
from typing import List

import torch

from catalyst.metrics.functional import process_recsys_components


def hitrate(outputs: torch.Tensor, targets: torch.Tensor, topk: List[int]) -> List[torch.Tensor]:
    """
    Calculate the hit rate score given model outputs and targets.
    Hit-rate is a metric for evaluating ranking systems.
    Generate top-N recommendations and if one of the recommendation is
    actually what user has rated, you consider that a hit.
    By rate we mean any explicit form of user's interactions.
    Add up all of the hits for all users and then divide by number of users

    Compute top-N recomendation for each user in the training stage
    and intentionally remove one of this items fro the training data.

    Args:
        outputs (torch.Tensor):
            Tensor weith predicted score
            size: [batch_size, slate_length]
            model outputs, logits
        targets (torch.Tensor):
            Binary tensor with ground truth.
            1 means the item is relevant
            for the user and 0 not relevant
            size: [batch_szie, slate_length]
            ground truth, labels
        topk (List[int]):
            Parameter fro evaluation on top-k items

    Returns:
        hitrate_at_k (List[torch.Tensor]):
            the hit rate score
    """
    results = []

    targets_sort_by_outputs = process_recsys_components(outputs, targets)
    for k in topk:
        k = min(outputs.size(1), k)
        hits_score = torch.sum(targets_sort_by_outputs[:, :k], dim=1) / k
        results.append(torch.mean(hits_score))

    return results


__all__ = ["hitrate"]
