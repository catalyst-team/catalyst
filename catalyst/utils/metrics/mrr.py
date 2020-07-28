"""
MRR metric.
"""

import torch


def mrr(outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:

    """
    Calculate the MRR score given model ouptputs and targets
    Args:
        outputs [batch_size, slate_length] (torch.Tensor):
            model outputs, logits
        targets [batch_szie, slate_length] (torch.Tensor):
            ground truth, labels
    
    Returns:
        mrr (float): the mrr score for each slate
    """
    max_rank = targets.shape[0]

    _, indices_for_sort = outputs.sort(descending=True, dim=-1)
    true_sorted_by_preds = torch.gather(
        targets, dim=-1, index=indices_for_sort
    )
    values, indices = torch.max(true_sorted_by_preds, dim=0)
    indices = indices.type_as(values).unsqueeze(dim=0).t()
    max_rank_rep = torch.tensor(
        data=max_rank, device=indices.device, dtype=torch.float32
    )
    within_at_mask = (indices < max_rank_rep).type(torch.float32)

    result = torch.tensor(1.0) / (indices + torch.tensor(1.0))

    zero_sum_mask = torch.sum(values) == 0.0
    result[zero_sum_mask] = 0.0

    mrr = result * within_at_mask
    return mrr


__all__ = ["mrr"]
