from typing import Optional, Union

import torch

from catalyst.metrics import precision_recall_fbeta_support
from catalyst.metrics.functional import process_multilabel_components


def average_precision(
    outputs: torch.Tensor, targets: torch.Tensor, weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Computes the average precision.

    Args:
        outputs: NxK tensor that for each of the N examples
            indicates the probability of the example belonging to each of
            the K classes, according to the model.
        targets:  binary NxK tensort that encodes which of the K
            classes are associated with the N-th input
            (eg: a row [0, 1, 0, 1] indicates that the example is
            associated with classes 2 and 4)
        weights: importance for each sample

    Returns:
        torch.Tensor: tensor of [K; ] shape,
        with average precision for K classes

    Examples:
        >>> average_precision(
        >>>     outputs=torch.Tensor([0.1, 0.4, 0.35, 0.8]),
        >>>     targets=torch.Tensor([0, 0, 1, 1]),
        >>> )
        tensor([0.8333])
    """
    # outputs - [bs; num_classes] with scores
    # targets - [bs; num_classes] with binary labels
    outputs, targets, weights = process_multilabel_components(
        outputs=outputs, targets=targets, weights=weights,
    )
    if outputs.numel() == 0:
        return torch.zeros(1)

    ap = torch.zeros(targets.size(1))

    # compute average precision for each class
    for class_i in range(targets.size(1)):
        # sort scores
        class_scores = outputs[:, class_i]
        class_targets = targets[:, class_i]
        _, sortind = torch.sort(class_scores, dim=0, descending=True)
        correct = class_targets[sortind]

        # compute true positive sums
        if weights is not None:
            class_weight = weights[sortind]
            weighted_correct = correct.float() * class_weight

            tp = weighted_correct.cumsum(0)
            rg = class_weight.cumsum(0)
        else:
            tp = correct.float().cumsum(0)
            rg = torch.arange(1, targets.size(0) + 1).float()

        # compute precision curve
        precision = tp.div(rg)

        # compute average precision
        ap[class_i] = precision[correct.bool()].sum() / max(float(correct.sum()), 1)

    return ap


def precision(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    argmax_dim: int = -1,
    eps: float = 1e-7,
    num_classes: Optional[int] = None,
) -> Union[float, torch.Tensor]:
    """
    Multiclass precision metric.

    Args:
        outputs: estimated targets as predicted by a model
            with shape [bs; ..., (num_classes or 1)]
        targets: ground truth (correct) target values
            with shape [bs; ..., 1]
        argmax_dim: int, that specifies dimension for argmax transformation
            in case of scores/probabilities in ``outputs``
        eps: float. Epsilon to avoid zero division.
        num_classes: int, that specifies number of classes if it known

    Returns:
        Tensor:

    Examples:
        >>> precision(
        >>>     outputs=torch.tensor([
        >>>         [1, 0, 0],
        >>>         [0, 1, 0],
        >>>         [0, 0, 1],
        >>>     ]),
        >>>     targets=torch.tensor([0, 1, 2]),
        >>>     beta=1,
        >>> )
        tensor([1., 1., 1.])
        >>> precision(
        >>>     outputs=torch.tensor([[0, 0, 1, 1, 0, 1, 0, 1]]),
        >>>     targets=torch.tensor([[0, 1, 0, 1, 0, 0, 1, 1]]),
        >>> )
        tensor([0.5000, 0.5000]
    """
    precision_score, _, _, _, = precision_recall_fbeta_support(
        outputs=outputs, targets=targets, argmax_dim=argmax_dim, eps=eps, num_classes=num_classes,
    )
    return precision_score


__all__ = ["average_precision", "precision"]
