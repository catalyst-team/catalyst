import torch


# @TODO: make it work in "per class" mode
def iou(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    eps: float = 1e-7,
    threshold: float = None,
) -> torch.Tensor:
    """Computes the dice score.

    Args:
        outputs: A list of predicted elements
        targets:  A list of elements that are to be predicted
        eps: epsilon to avoid zero division
        threshold: threshold for outputs binarization

    Returns:
        IoU (Jaccard) score
    """
    if threshold is not None:
        outputs = (outputs > threshold).float()

    intersection = torch.sum(targets * outputs)
    union = torch.sum(targets) + torch.sum(outputs)
    # this looks a bit awkward but `eps * (union == 0)` term
    # makes sure that if I and U are both 0, than IoU == 1
    # and if U != 0 and I == 0 the eps term in numerator is zeroed out
    # i.e. (0 + eps) / (U - 0 + eps) doesn't happen
    iou_score = (intersection + eps * (union == 0)) / (
        union - intersection + eps
    )

    return iou_score


jaccard = iou

__all__ = ["iou", "jaccard"]
