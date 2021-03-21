"""
Hitrate metric:
    * :func:`hitrate`
"""
import torch


def hitrate(outputs: torch.Tensor, targets: torch.Tensor):
    """
    Calculate the hit rate score given model outputs and targets
    Args:
        outputs (torch.Tensor): model outputs, logits
        targets (torch.Tensor): ground truth, labels
    Returns:
        hitrate (torch.Tensor): the hit rate score
    """
    outputs = outputs.clone()
    targets = targets.clone()

    targets_expanded = targets.view(-1, 1)
    targets_expanded = targets_expanded.expand_as(outputs)
    row_hits, _ = torch.max((targets_expanded == outputs), dim=1)
    hits = row_hits.float().mean()
    return hits


__all__ = ["hitrate"]
