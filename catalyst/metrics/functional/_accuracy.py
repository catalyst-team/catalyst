from typing import Sequence, Union

import numpy as np
import torch

from catalyst.metrics.functional import process_multilabel_components


def accuracy(
    outputs: torch.Tensor, targets: torch.Tensor, topk: Sequence[int] = (1,),
) -> Sequence[torch.Tensor]:
    """
    Computes multiclass accuracy@topk for the specified values of `topk`.

    Args:
        outputs: model outputs, logits
            with shape [bs; num_classes]
        targets: ground truth, labels
            with shape [bs; 1]
        topk: `topk` for accuracy@topk computing

    Returns:
        list with computed accuracy@topk

    Example:
        >>> accuracy(
        >>>     outputs=torch.tensor([
        >>>         [1, 0, 0],
        >>>         [0, 1, 0],
        >>>         [0, 0, 1],
        >>>     ]),
        >>>     targets=torch.tensor([0, 1, 2]),
        >>> )
        [tensor([1.])]
        >>> accuracy(
        >>>     outputs=torch.tensor([
        >>>         [1, 0, 0],
        >>>         [0, 1, 0],
        >>>         [0, 1, 0],
        >>>     ]),
        >>>     targets=torch.tensor([0, 1, 2]),
        >>> )
        [tensor([0.6667])]
        >>> accuracy(
        >>>     outputs=torch.tensor([
        >>>         [1, 0, 0],
        >>>         [0, 1, 0],
        >>>         [0, 0, 1],
        >>>     ]),
        >>>     targets=torch.tensor([0, 1, 2]),
        >>>     topk=[1, 3],
        >>> )
        [tensor([1.]), tensor([1.])]
        >>> accuracy(
        >>>     outputs=torch.tensor([
        >>>         [1, 0, 0],
        >>>         [0, 1, 0],
        >>>         [0, 1, 0],
        >>>     ]),
        >>>     targets=torch.tensor([0, 1, 2]),
        >>>     topk=[1, 3],
        >>> )
        [tensor([0.6667]), tensor([1.])]
    """
    max_k = max(topk)
    batch_size = targets.size(0)

    if len(outputs.shape) == 1 or outputs.shape[1] == 1:
        # binary accuracy
        pred = outputs.t()
    else:
        # multiclass accuracy
        _, pred = outputs.topk(max_k, 1, True, True)  # noqa: WPS425
        pred = pred.t()
    correct = pred.eq(targets.long().view(1, -1).expand_as(pred))

    output = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
        output.append(correct_k.mul_(1.0 / batch_size))
    return output


def multilabel_accuracy(
    outputs: torch.Tensor, targets: torch.Tensor, threshold: Union[float, torch.Tensor],
) -> torch.Tensor:
    """
    Computes multilabel accuracy for the specified activation and threshold.

    Args:
        outputs: NxK tensor that for each of the N examples
            indicates the probability of the example belonging to each of
            the K classes, according to the model.
        targets: binary NxK tensort that encodes which of the K
            classes are associated with the N-th input
            (eg: a row [0, 1, 0, 1] indicates that the example is
            associated with classes 2 and 4)
        threshold: threshold for for model output

    Returns:
        computed multilabel accuracy

    Example:
        >>> multilabel_accuracy(
        >>>     outputs=torch.tensor([
        >>>         [1, 0],
        >>>         [0, 1],
        >>>     ]),
        >>>     targets=torch.tensor([
        >>>         [1, 0],
        >>>         [0, 1],
        >>>     ]),
        >>>     threshold=0.5,
        >>> )
        tensor([1.])
        >>> multilabel_accuracy(
        >>>     outputs=torch.tensor([
        >>>         [1.0, 0.0],
        >>>         [0.6, 1.0],
        >>>     ]),
        >>>     targets=torch.tensor([
        >>>         [1, 0],
        >>>         [0, 1],
        >>>     ]),
        >>>     threshold=0.5,
        >>> )
        tensor(0.7500)
        >>> multilabel_accuracy(
        >>>     outputs=torch.tensor([
        >>>         [1.0, 0.0],
        >>>         [0.4, 1.0],
        >>>     ]),
        >>>     targets=torch.tensor([
        >>>         [1, 0],
        >>>         [0, 1],
        >>>     ]),
        >>>     threshold=0.5,
        >>> )
        tensor(1.0)
    """
    outputs, targets, _ = process_multilabel_components(outputs=outputs, targets=targets)

    outputs = (outputs > threshold).long()
    output = (targets.long() == outputs.long()).sum().float() / np.prod(targets.shape)
    return output


__all__ = ["accuracy", "multilabel_accuracy"]
