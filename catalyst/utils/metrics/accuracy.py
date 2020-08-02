"""
Various accuracy metrics:
    * :func:`accuracy`
    * :func:`multi_label_accuracy`
"""
from typing import Optional, Sequence, Union

import numpy as np

import torch

from catalyst.utils.metrics.functional import preprocess_multi_label_metrics
from catalyst.utils.torch import get_activation_fn


def accuracy(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    topk: Sequence[int] = (1,),
    activation: Optional[str] = None,
) -> Sequence[torch.Tensor]:
    """
    Computes multi-class accuracy@topk for the specified values of `topk`.

    Args:
        outputs (torch.Tensor): model outputs, logits
            with shape [bs; num_classes]
        targets (torch.Tensor): ground truth, labels
            with shape [bs; 1]
        activation (str): activation to use for model output
        topk (Sequence[int]): `topk` for accuracy@topk computing

    Returns:
        list with computed accuracy@topk
    """
    activation_fn = get_activation_fn(activation)
    outputs = activation_fn(outputs)

    max_k = max(topk)
    batch_size = targets.size(0)

    if len(outputs.shape) == 1 or outputs.shape[1] == 1:
        # binary accuracy
        pred = outputs.t()
    else:
        # multi-class accuracy
        _, pred = outputs.topk(max_k, 1, True, True)  # noqa: WPS425
        pred = pred.t()
    correct = pred.eq(targets.long().view(1, -1).expand_as(pred))

    output = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        output.append(correct_k.mul_(1.0 / batch_size))
    return output


def multi_label_accuracy(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    threshold: Union[float, torch.Tensor],
    activation: Optional[str] = None,
) -> torch.Tensor:
    """
    Computes multi-label accuracy for the specified activation and threshold.

    Args:
        outputs (torch.Tensor): NxK tensor that for each of the N examples
            indicates the probability of the example belonging to each of
            the K classes, according to the model.
        targets (torch.Tensor): binary NxK tensort that encodes which of the K
            classes are associated with the N-th input
            (eg: a row [0, 1, 0, 1] indicates that the example is
            associated with classes 2 and 4)
        threshold (float): threshold for for model output
        activation (str): activation to use for model output

    Returns:
        computed multi-label accuracy
    """
    outputs, targets, _ = preprocess_multi_label_metrics(
        outputs=outputs, targets=targets
    )
    activation_fn = get_activation_fn(activation)
    outputs = activation_fn(outputs)

    outputs = (outputs > threshold).long()
    output = (targets.long() == outputs.long()).sum().float() / np.prod(
        targets.shape
    )
    return output


__all__ = ["accuracy", "multi_label_accuracy"]
