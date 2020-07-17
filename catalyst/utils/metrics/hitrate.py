"""
Hitrate metric
    * :func:`hitrate`
"""
import torch


def hitrate(outputs: torch.Tensor, targets: torch.Tensor):
    """
    Calculate the hit rate score given model outputs and targets.

    Args:
        outputs (torch.Tensor): Model outputs, logits
        targets (torch.Tensor): Ground truth, labels
    Returns:
        hit rate (torch.Tensor): The hit rate
    """
    outputs = outputs.clone()
    targets = targets.clone()

    targets_expanded = targets.view(-1, 1)
    targets_expanded = targets_expanded.expand_as(outputs)
    row_hits, _ = torch.max((targets_expanded == outputs), dim=1)
    hits = row_hits.float().mean()
    return hits


__all__ = ["hitrate"]
