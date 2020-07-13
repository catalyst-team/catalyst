"""
Cumulative Gain metrics:
    * :func:`dcg`
    * :func:`ndcg`
"""
import torch


def dcg(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    k: int = 10,
    gains: str = "linear",
):
    """
    Computes the discounted cumulative gain (DCG) at k.

    Args:
        outputs (torch.Tensor): model outputs, logits
        targets (torch.Tensor): ground truth, labels
        k (int, optional): the maximum number of predicted elements
        gains (str): indicates whether gains are "linear" (default) or "exp"

    Returns: # noqa: DAR201
        float: computed DCG@k

    Raises:
        ValueError: If `gains` not in ["linear", "exp"].
    """
    targets, _ = targets.topk(k, -1)
    outputs, _ = outputs.topk(k, -1)

    if gains == "linear":
        gains = targets
    elif gains == "exp":
        gains = 2 ** targets - 1
    else:
        raise ValueError("No such gains option.")

    discounts = torch.log2(torch.arange(len(targets)) + 2.0)
    dcg_score = float(torch.sum(gains / discounts))
    return dcg_score


def ndcg(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    k: int = 10,
    gains: str = "linear",
):
    """
    Computes the normalized discounted cumulative gain (DCG) at k.

    Args:
        outputs (torch.Tensor): model outputs, logits
        targets (torch.Tensor): ground truth, labels
        k (int, optional): The maximum number of predicted elements
        gains (str): indicates whether gains are "exp" (default) or "linear"

    Returns: # noqa: DAR201
        float: computed nDCG@k
    """
    ideal = dcg(targets, targets, k, gains)
    actual = dcg(outputs, targets, k, gains)
    if best == 0:
        ndcg_score = 0
    else:
        ndcg_score = actual / best
    return ndcg_score


__all__ = ["dcg", "ndcg"]
