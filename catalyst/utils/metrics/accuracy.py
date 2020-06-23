"""
Various accuracy metrics:
    * :func:`accuracy`
    * :func:`average_accuracy`
    * :func:`mean_average_accuracy`
"""
from typing import Tuple

import numpy as np

import torch

from catalyst.utils.torch import get_activation_fn


def accuracy(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    topk: Tuple = (1,),
    threshold: float = None,
    activation: str = None,
):
    """
    Computes the accuracy.

    It can be used either for:

    1. Multi-class task, in this case:

      - you can use topk.
      - threshold and activation are not required.
      - targets is a tensor: batch_size
      - outputs is a tensor: batch_size x num_classes
      - computes the accuracy@k for the specified values of k.

    2. Multi-label task, in this case:

      - you must specify threshold and activation
      - topk will not be used
        (because of there is no method to apply top-k in
        multi-label classification).
      - outputs, targets are tensors with shape: batch_size x num_classes
      - targets is a tensor with binary vectors

    Args:
        outputs (torch.Tensor): model outputs, logits
        targets (torch.Tensor): ground truth, labels
        topk (tuple): tuple with specified `N` for top`N` accuracy computing
        threshold (float): threshold for outputs
        activation (str): activation for outputs

    Returns:
        computed topK accuracy
    """
    activation_fn = get_activation_fn(activation)
    outputs = activation_fn(outputs)

    if threshold:
        outputs = (outputs > threshold).long()

    # TODO: move to separate function
    # multi-label classification
    if len(targets.shape) > 1 and targets.size(1) > 1:
        output = (targets.long() == outputs.long()).sum().float() / np.prod(
            targets.shape
        )
        return [output]

    max_k = max(topk)
    batch_size = targets.size(0)

    if len(outputs.shape) == 1 or outputs.shape[1] == 1:
        pred = outputs.t()
    else:
        _, pred = outputs.topk(max_k, 1, True, True)  # noqa: WPS425
        pred = pred.t()
    correct = pred.eq(targets.long().view(1, -1).expand_as(pred))

    output = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        output.append(correct_k.mul_(1.0 / batch_size))
    return output


def average_accuracy(outputs, targets, k=10):
    """Computes the average accuracy at k.

    This function computes the average
    accuracy at k between two lists of items.

    Args:
        outputs (list): A list of predicted elements
        targets (list):  A list of elements that are to be predicted
        k (int, optional): The maximum number of predicted elements

    Returns:
        float: The average accuracy at k over the input lists
    """
    if len(outputs) > k:
        outputs = outputs[:k]

    score = 0.0
    num_hits = 0.0

    for i, predict in enumerate(outputs):
        if predict in targets and predict not in outputs[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    if not targets:
        return 0.0

    return score / min(len(targets), k)


def mean_average_accuracy(outputs, targets, topk=(1,)):
    """Computes the mean average accuracy at k.

    This function computes the mean average accuracy at k between two lists
    of lists of items.

    Args:
        outputs (list): A list of lists of predicted elements
        targets (list): A list of lists of elements that are to be predicted
        topk (int, optional): The maximum number of predicted elements

    Returns:
        float: The mean average accuracy at k over the input lists
    """
    max_k = max(topk)
    _, pred = outputs.topk(max_k, 1, True, True)  # noqa: WPS425

    targets = targets.data.cpu().numpy().tolist()
    actual_list = []
    for a in targets:
        actual_list.append([a])
    targets = actual_list
    pred = pred.tolist()

    res = []
    for k in topk:
        ap = np.mean(
            [average_accuracy(p, a, k) for a, p in zip(targets, pred)]
        )
        res.append(ap)
    return res


__all__ = ["accuracy", "average_accuracy", "mean_average_accuracy"]
