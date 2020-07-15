"""
Various accuracy metrics:
    * :func:`accuracy`
    * :func:`average_accuracy`
    * :func:`mean_average_accuracy`
"""
from typing import List, Tuple

import numpy as np

import torch

from catalyst.utils.torch import get_activation_fn


def multi_class_accuracy(
    outputs: torch.Tensor, targets: torch.Tensor, topk: Tuple = (1,),
) -> List:
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
    threshold: float = torch.Tensor,
) -> torch.Tensor:
    outputs = (outputs > threshold).long()
    output = (targets.long() == outputs.long()).sum().float() / np.prod(
        targets.shape
    )
    return output


def accuracy(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    activation: str = None,
    multi_label: bool = False,
    topk: Tuple = (1,),
    threshold: float = None,
) -> List:
    """
    Computes the accuracy.

    It can be used either for:

    1. Multi-class task (`multi_label=False`), in this case:

      - you can use topk.
      - threshold and activation are not required.
      - targets is a tensor: batch_size
      - outputs is a tensor: batch_size x num_classes
      - computes the accuracy@k for the specified values of k.

    2. Multi-label task (`multi_label=True`), in this case:

      - you must specify threshold and activation
      - topk will not be used
        (because of there is no method to apply top-k in
        multi-label classification).
      - outputs, targets are tensors with shape: batch_size x num_classes
      - targets is a tensor with binary vectors

    Args:
        outputs (torch.Tensor): model outputs, logits
        targets (torch.Tensor): ground truth, labels
        activation (str): activation for outputs
        multi_label (bool): boolean flag to compute multi-label case
        topk (tuple): tuple with specified `N` for top`N` accuracy computing
        threshold (float): threshold for outputs

    Returns:
        computed accuracy
    """
    activation_fn = get_activation_fn(activation)
    outputs = activation_fn(outputs)

    if multi_label:
        output = multi_label_accuracy(
            outputs=outputs, targets=targets, threshold=threshold
        )
        return [output]
    else:
        return multi_class_accuracy(
            outputs=outputs, targets=targets, topk=topk
        )


__all__ = ["accuracy", "multi_label_accuracy", "multi_class_accuracy"]
