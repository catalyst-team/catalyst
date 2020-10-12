from typing import Callable, Dict, Optional, Sequence, Tuple
from functools import partial

import torch
from torch import Tensor
from torch.nn import functional as F


def get_binary_statistics(
    predictions: Tensor, targets: Tensor
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """
    Computes the number of true positive, false positive, true negative,
    false negative and support for a binary classification problem.

    Args:
        predictions: Estimated targets as predicted by a model.
        targets: Ground truth (correct) target values.

    Returns:
        Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]: stats
    """
    true_positive = ((predictions == 1) * (targets == 1)).to(torch.long).sum()
    false_positive = ((predictions == 1) * (targets != 1)).to(torch.long).sum()
    true_negative = ((predictions != 1) * (targets != 1)).to(torch.long).sum()
    false_negative = ((predictions != 1) * (targets == 1)).to(torch.long).sum()
    support = (targets == 1).to(torch.long).sum()

    return (
        true_positive,
        false_positive,
        true_negative,
        false_negative,
        support,
    )


def preprocess_multi_label_metrics(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    General preprocessing and check for multi-label-based metrics.

    Args:
        outputs: NxK tensor that for each of the N examples
            indicates the probability of the example belonging to each of
            the K classes, according to the model.
        targets: binary NxK tensor that encodes which of the K
            classes are associated with the N-th input
            (eg: a row [0, 1, 0, 1] indicates that the example is
            associated with classes 2 and 4)
        weights: importance for each sample

    Returns:
        processed ``outputs`` and ``targets``
        with [batch_size; num_classes] shape
    """
    if not torch.is_tensor(outputs):
        outputs = torch.from_numpy(outputs)
    if not torch.is_tensor(targets):
        targets = torch.from_numpy(targets)
    if weights is not None:
        if not torch.is_tensor(weights):
            weights = torch.from_numpy(weights)
        weights = weights.squeeze()

    if outputs.dim() == 1:
        outputs = outputs.view(-1, 1)
    else:
        assert outputs.dim() == 2, (
            "wrong `outputs` size "
            "(should be 1D or 2D with one column per class)"
        )

    if targets.dim() == 1:
        if outputs.shape[1] > 1:
            # multi-class case
            num_classes = outputs.shape[1]
            targets = F.one_hot(targets, num_classes).float()
        else:
            # binary case
            targets = targets.view(-1, 1)
    else:
        assert targets.dim() == 2, (
            "wrong `targets` size "
            "(should be 1D or 2D with one column per class)"
        )

    if weights is not None:
        assert weights.dim() == 1, "Weights dimension should be 1"
        assert weights.numel() == targets.size(
            0
        ), "Weights dimension 1 should be the same as that of target"
        assert torch.min(weights) >= 0, "Weight should be non-negative only"

    assert torch.equal(
        targets ** 2, targets
    ), "targets should be binary (0 or 1)"

    return outputs, targets, weights


def get_default_topk_args(num_classes: int) -> Sequence[int]:
    """Calculate list params for ``Accuracy@k`` and ``mAP@k``.

    Examples:
        >>> get_default_topk_args(num_classes=4)
        >>> [1, 3]
        >>> get_default_topk_args(num_classes=8)
        >>> [1, 3, 5]

    Args:
        num_classes: number of classes

    Returns:
        iterable: array of accuracy arguments
    """
    result = [1]

    if num_classes is None:
        return result

    if num_classes > 3:
        result.append(3)
    if num_classes > 5:
        result.append(5)

    return result


def wrap_class_metric2dict(
    metric_fn: Callable, class_args: Sequence[str] = None
) -> Callable:
    """# noqa: D202
    Logging wrapper for metrics with torch.Tensor output
    and [num_classes] shape.
    Computes the metric and sync each element from the output Tensor
    with passed `class` argument.

    Args:
        metric_fn: metric function to compute
        class_args: class names for logging.
            default: None - class indexes will be used.

    Returns:
        wrapped metric function with List[Dict] output
    """

    def class_metric_with_dict_output(*args, **kwargs):
        output = metric_fn(*args, **kwargs)
        num_classes = len(output)
        output_class_args = class_args or [
            f"/class_{i:02}" for i in range(num_classes)
        ]
        mean_stats = torch.mean(output).item()
        output = {
            key: value.item() for key, value in zip(output_class_args, output)
        }
        output["/mean"] = mean_stats
        return output

    return class_metric_with_dict_output


def wrap_topk_metric2dict(
    metric_fn: Callable, topk_args: Sequence[int]
) -> Callable:
    """
    Logging wrapper for metrics with
    Sequence[Union[torch.Tensor, int, float, Dict]] output.
    Computes the metric and sync each element from the output sequence
    with passed `topk` argument.

    Args:
        metric_fn: metric function to compute
        topk_args: topk args to sync outputs with

    Returns:
        wrapped metric function with List[Dict] output

    Raises:
        NotImplementedError: if metrics returned values are out of
            torch.Tensor, int, float, Dict union.

    """
    metric_fn = partial(metric_fn, topk=topk_args)

    def topk_metric_with_dict_output(*args, **kwargs):
        output: Sequence = metric_fn(*args, **kwargs)

        if isinstance(output[0], (int, float, torch.Tensor)):
            output = {
                f"{topk_key:02}": metric_value
                for topk_key, metric_value in zip(topk_args, output)
            }
        elif isinstance(output[0], Dict):
            output = {
                {
                    f"{metric_key}{topk_key:02}": metric_value
                    for metric_key, metric_value in metric_dict_value.items()
                }
                for topk_key, metric_dict_value in zip(topk_args, output)
            }
        else:
            raise NotImplementedError()

        return output

    return topk_metric_with_dict_output


__all__ = [
    "get_binary_statistics",
    "wrap_topk_metric2dict",
    "wrap_class_metric2dict",
]
