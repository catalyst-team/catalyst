from typing import Optional, Sequence, Tuple

import torch


def preprocess_multi_label_metrics(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    General preprocessing and check for multi-label-based metrics.

    Args:
        outputs (torch.Tensor): NxK tensor that for each of the N examples
            indicates the probability of the example belonging to each of
            the K classes, according to the model.
        targets (torch.Tensor): binary NxK tensor that encodes which of the K
            classes are associated with the N-th input
            (eg: a row [0, 1, 0, 1] indicates that the example is
            associated with classes 2 and 4)
        weights (torch.Tensor): importance for each sample

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
        num_classes (int): number of classes

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


def wrap_topk_metric2dict(metric_fn, topk_args):
    def topk_metric_with_dict_output(*args, **kwargs):
        metric_output = metric_fn(*args, **kwargs, topk=topk_args)
        metric_output = {
            f"{key:02}": value for key, value in zip(topk_args, metric_output)
        }
        return metric_output

    return topk_metric_with_dict_output


def wrap_class_metric2dict(metric_fn, class_args):
    def class_metric_with_dict_output(*args, **kwargs):
        metric_output = metric_fn(*args, **kwargs)
        # metric_output = {
        #     f"{key:02}": value for key, value in zip(topk, metric_output)
        # }
        return metric_output

    return class_metric_with_dict_output
