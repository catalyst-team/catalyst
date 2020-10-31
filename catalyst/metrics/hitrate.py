"""
Hitrate metric:
    * :func:`hitrate`
"""
import torch


def hitrate(
    outputs: torch.Tensor, targets: torch.Tensor, k=10
) -> torch.Tensor:
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
        k (int):
            Parameter fro evaluation on top-k items

    Returns:
        hitrate (torch.Tensor): the hit rate score
    """
    k = min(outputs.size(1), k)

    _, indices_for_sort = outputs.sort(descending=True, dim=-1)
    true_sorted_by_preds = torch.gather(
        targets, dim=-1, index=indices_for_sort
    )
    true_sorted_by_pred_shrink = true_sorted_by_preds[:, :k]
    hits = torch.sum(true_sorted_by_pred_shrink, dim=1) / k
    return hits


__all__ = ["hitrate"]
