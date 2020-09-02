"""
MRR metric.
"""

import torch


def mrr(outputs: torch.Tensor, targets: torch.Tensor, k=100) -> torch.Tensor:

    """
    Calculate the MRR score given model ouptputs and targets
    Args:
        outputs (torch.Tensor):
            size: [batch_size, slate_length] 
            model outputs, logits
        targets (torch.Tensor):
            size: [batch_szie, slate_length]
            ground truth, labels

    Returns:
        mrr (float): the mrr score for each slate
    """
    max_rank = targets.shape[0]
    # print(targets.size())
    # if len(targets.size()) > 2: 
    #     k = min(targets.size()[1], k)

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
    # print(result*within_at_mask)
    return result * within_at_mask


__all__ = ["mrr"]
