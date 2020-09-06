"""
MRR metric.
"""

import torch


def mrr(outputs: torch.Tensor, targets: torch.Tensor, k=100) -> torch.Tensor:

    """
    Calculate the MRR score given model ouptputs and targets
    Users data represnted via batches.
    Args:
        outputs (torch.Tensor):
            Tensor weith predicted score
            size: [batch_size, slate_length] 
            model outputs, logits
        targets (torch.Tensor):
            Binary tensor with ground truth. 1 means the item is relevant
            for the user and 0 not relevant
            size: [batch_szie, slate_length]
            ground truth, labels

    Returns:
        mrr (float): the mrr score for each slate
    """
    k = min(outputs.size()[1], k)
    _, indices_for_sort = outputs.sort(descending=True, dim=-1)
    true_sorted_by_preds = torch.gather(
        targets, dim=-1, index=indices_for_sort
    )
    true_sorted_by_pred_shrink = true_sorted_by_preds[:, :k]

    values, indices = torch.max(true_sorted_by_pred_shrink, dim=1)
    indices = indices.type_as(values).unsqueeze(dim=0).t()
    result = torch.tensor(1.0) / (indices + torch.tensor(1.0))

    zero_sum_mask = values == 0.0
    result[zero_sum_mask] = 0.0
    return result


__all__ = ["mrr"]
