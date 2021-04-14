from typing import List

import torch

from catalyst.metrics.functional._misc import process_recsys_components


def _nan_to_num(tensor, nan=0.0):
    tensor = torch.where(torch.isnan(tensor), torch.ones_like(tensor) * nan, tensor)
    return tensor


# nan_to_num is available in PyTorch only from 1.8.0 version
NAN_TO_NUM_FN = torch.__dict__.get("nan_to_num", _nan_to_num)


def hitrate(
    outputs: torch.Tensor, targets: torch.Tensor, topk: List[int], zero_division: int = 0
) -> List[torch.Tensor]:
    """
    Calculate the hit rate (aka recall) score given
    model outputs and targets.
    Hit-rate is a metric for evaluating ranking systems.
    Generate top-N recommendations and if one of the recommendation is
    actually what user has rated, you consider that a hit.
    By rate we mean any explicit form of user's interactions.
    Add up all of the hits for all users and then divide by number of users

    Compute top-N recommendation for each user in the training stage
    and intentionally remove one of this items fro the training data.

    Args:
        outputs (torch.Tensor):
            Tensor with predicted score
            size: [batch_size, slate_length]
            model outputs, logits
        targets (torch.Tensor):
            Binary tensor with ground truth.
            1 means the item is relevant
            for the user and 0 not relevant
            size: [batch_size, slate_length]
            ground truth, labels
        topk (List[int]):
            Parameter fro evaluation on top-k items
        zero_division (int):
            value, returns in the case of the divison by zero
            should be one of 0 or 1

    Returns:
        hitrate_at_k (List[torch.Tensor]): the hitrate score
    """
    results = []

    targets_sort_by_outputs = process_recsys_components(outputs, targets)
    for k in topk:
        k = min(outputs.size(1), k)
        hits_score = torch.sum(targets_sort_by_outputs[:, :k], dim=1) / targets.sum(dim=1)
        hits_score = NAN_TO_NUM_FN(hits_score, zero_division)
        results.append(torch.mean(hits_score))

    return results


__all__ = ["hitrate"]
