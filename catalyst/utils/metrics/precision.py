from typing import Optional, Sequence

import torch

from catalyst.utils.metrics.functional import preprocess_multi_label_metrics


def average_precision(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
    topk: Sequence[int] = (1,),
) -> Sequence[torch.Tensor]:
    """Computes the average precision at `topk`.

    Args:
        outputs (torch.Tensor): NxK tensor that for each of the N examples
            indicates the probability of the example belonging to each of
            the K classes, according to the model.
        targets (torch.Tensor):  binary NxK tensort that encodes which of the K
            classes are associated with the N-th input
            (eg: a row [0, 1, 0, 1] indicates that the example is
            associated with classes 2 and 4)
        weights (torch.Tensor): importance for each sample
        topk (int, optional): The maximum number of predicted elements

    Returns:
        Sequence[torch.Tensor]: list of 1xK tensor,
        with average precision@topk_i for K classes
    """
    assert len(topk) == 1 and topk[0] == 1, "@K logic is not implemented yet"
    # outputs - [bs; num_classes] with scores
    # targets - [bs; num_classes] with binary labels
    outputs, targets, weights = preprocess_multi_label_metrics(
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
        ap[class_i] = precision[correct.bool()].sum() / max(
            float(correct.sum()), 1
        )

    return [ap]


def mean_average_precision(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
    topk: Sequence[int] = (1,),
) -> Sequence[torch.Tensor]:
    """Computes the mean average precision at `topk`.

    Args:
        outputs (torch.Tensor): NxK tensor that for each of the N examples
            indicates the probability of the example belonging to each of
            the K classes, according to the model.
        targets (torch.Tensor): binary NxK tensort that encodes which of the K
            classes are associated with the N-th input
            (eg: a row [0, 1, 0, 1] indicates that the example is
            associated with classes 2 and 4)
        weights (torch.Tensor): importance for each sample
        topk (int, optional): The maximum number of predicted elements

    Returns:
        Sequence[torch.Tensor]: list of 1x1 tensor,
        with mean average precision@topk_i for K classes
    """
    output = average_precision(
        outputs=outputs, targets=targets, weights=weights, topk=topk
    )
    output = [x.mean() for x in output]
    return output


__all__ = ["average_precision", "mean_average_precision"]
