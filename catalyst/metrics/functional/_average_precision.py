from typing import List, Optional

import torch

from catalyst.metrics.functional._misc import (
    process_multilabel_components,
    process_recsys_components,
)


def binary_average_precision(
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
        >>> binary_average_precision(
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


def average_precision(outputs: torch.Tensor, targets: torch.Tensor, k: int) -> torch.Tensor:
    """
    Calculate the Average Precision for RecSys.
    The precision metric summarizes the fraction of relevant items
    out of the whole the recommendation list.

    To compute the precision at k set the threshold rank k,
    compute the percentage of relevant items in topK,
    ignoring the documents ranked lower than k.

    The average precision at k (AP at k) summarizes the average
    precision for relevant items up to the k-th one.
    Wikipedia entry for the Average precision

    <https://en.wikipedia.org/w/index.php?title=Information_retrieval&
    oldid=793358396#Average_precision>

    If a relevant document never gets retrieved,
    we assume the precision corresponding to that
    relevant doc to be zero

    Args:
        outputs (torch.Tensor):
            Tensor with predicted score
            size: [batch_size, slate_length]
            model outputs, logits
        targets (torch.Tensor):
            Binary tensor with ground truth.
            1 means the item is relevant
            and 0 not relevant
            size: [batch_szie, slate_length]
            ground truth, labels
        k:
            Parameter for evaluation on top-k items

    Returns:
        ap_score (torch.Tensor):
            The map score for each batch.
            size: [batch_size, 1]

    Examples:
        >>> average_precision(
        >>>   outputs=torch.tensor([
        >>>     [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
        >>>     [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
        >>>   ]),
        >>>   targets=torch.tensor([
        >>>     [1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0],
        >>>     [0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        >>>   ]),
        >>> )
        tensor([0.6222, 0.4429])
    """
    targets_sort_by_outputs = process_recsys_components(outputs, targets)[:, :k]
    precisions = torch.zeros_like(targets_sort_by_outputs)

    for index in range(k):
        precisions[:, index] = torch.sum(targets_sort_by_outputs[:, : (index + 1)], dim=1) / float(
            index + 1
        )

        precisions[:, index] = torch.sum(targets_sort_by_outputs[:, : (index + 1)], dim=1) / float(
            index + 1
        )

    only_relevant_precision = precisions * targets_sort_by_outputs
    ap_score = only_relevant_precision.sum(dim=1) / ((only_relevant_precision != 0).sum(dim=1))
    ap_score[torch.isnan(ap_score)] = 0
    return ap_score


def mean_average_precision(
    outputs: torch.Tensor, targets: torch.Tensor, topk: List[int]
) -> List[torch.Tensor]:
    """
    Calculate the mean average precision (MAP) for RecSys.
    The metrics calculate the mean of the AP across all batches

    MAP amplifies the interest in finding many
    relevant items for each query

    Args:
        outputs (torch.Tensor): Tensor with predicted score
            size: [batch_size, slate_length]
            model outputs, logits
        targets (torch.Tensor):
            Binary tensor with ground truth.
            1 means the item is relevant and 0 not relevant
            size: [batch_szie, slate_length]
            ground truth, labels
        topk (List[int]):
            List of parameter for evaluation
            topK items

    Returns:
        map_at_k (Tuple[float]):
        The map score for every k.
        size: len(top_k)

    Examples:
        >>> mean_average_precision(
        >>>   outputs=torch.tensor([
        >>>     [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
        >>>     [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
        >>>   ]),
        >>>   targets=torch.tensor([
        >>>     [1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0],
        >>>     [0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        >>>   ]),
        >>>   topk=[10],
        >>> )
        [tensor(0.5325)]
    """
    results = []
    for k in topk:
        k = min(outputs.size(1), k)
        results.append(torch.mean(average_precision(outputs, targets, k)))

    return results


__all__ = [
    "binary_average_precision",
    "mean_average_precision",
    "average_precision",
]
