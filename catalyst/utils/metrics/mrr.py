"""
MRR metric.
"""

import torch


def mrr(outputs: torch.Tensor, targets: torch.Tensor):

    """
    Calculate the MRR score given model ouptputs and targets
    Args:
        outputs [batch_size, slate_length] (torch.Tensor): 
            model outputs, logits
        targets [batch_szie, slate_length] (torch.Tensor): 
            ground truth, labels
    Returns:
        mrr (float): the mrr score
    """
    max_rank = targets.shape[0]

    _, indices = outputs.sort(descending=True, dim=-1)
    true_sorted_by_preds = torch.gather(targets, dim=0, index=indices)
    values, indices = torch.max(true_sorted_by_preds, dim=0)
    indices = indices.type_as(values).unsqueeze(dim=0).t()
    ats_rep = torch.tensor(
        data=max_rank, device=indices.device, dtype=torch.float32
    )
    within_at_mask = (indices < ats_rep).type(torch.float32)

    result = torch.tensor(1.0) / (indices + torch.tensor(1.0))

    zero_sum_mask = torch.sum(values) == 0.0
    result[zero_sum_mask] = 0.0

    mrr = result * within_at_mask
    return mrr[0]


__all__ = ["mrr"]
