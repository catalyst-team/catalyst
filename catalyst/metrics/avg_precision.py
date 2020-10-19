"""
MAP metric.
"""

import torch


def avg_precision(outputs: torch.Tensor, targets: torch.Tensor, k=100) -> torch.Tensor:
    """
    Calculate the Mean Average Precision (MAP)
    The precision metric summarizes the fraction of relevant items
    out of the whole the recommendation list.
    The average precision @ k (AP@k) metrics summarizes the average
    precision achieveid in every item up to k-th one.
    The mean average precision calcultaes the mean over all users

    Example: targets labels [1,1,0], k = 3
    Precsion@k:
    [1/1, 2/2, 2/3]
    Average Precision@k:
    (1/3)[1 + 1 + 2/3] = 0.88

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
        result (torch.Tensor):
            The mrr score for each user.

    References:
    `Wikipedia entry for the Average precision
           <https://en.wikipedia.org/w/index.php?title=Information_retrieval&
           oldid=793358396#Average_precision>`
    """

    k =  min(outputs.size(1), k)
    _, indices_for_sort = outputs.sort(descending=True, dim=-1)
    true_sorted_by_preds = torch.gather(
        targets, dim=-1, index=indices_for_sort
    )

    true_sorted_by_pred_topk = true_sorted_by_preds[:, :k]
    precisions = torch.zeros_like(true_sorted_by_pred_topk)

    for index in range(k):
        precisions[:, index] = torch.sum(true_sorted_by_pred_topk[:, : (index+1)], dim=1) / float(index+1)

    ap = torch.mean(precisions, dim=1)
    return ap


__all__ = ["avg_precision"]
