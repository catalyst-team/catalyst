from typing import Optional, Sequence, Tuple
import logging

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

logger = logging.getLogger(__name__)


def process_multiclass_components(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    argmax_dim: int = -1,
    num_classes: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """
    Preprocess input in case multiclass classification task.

    Args:
        outputs: estimated targets as predicted by a model
            with shape [bs; ..., (num_classes or 1)]
        targets: ground truth (correct) target values
            with shape [bs; ..., 1]
        argmax_dim: int, that specifies dimension for argmax transformation
            in case of scores/probabilities in ``outputs``
        num_classes: int, that specifies number of classes if it known

    Returns:
        preprocessed outputs, targets and num_classes
    """
    # @TODO: better multiclass preprocessing, label -> class_id mapping
    if not torch.is_tensor(outputs):
        outputs = torch.from_numpy(np.array(outputs))
    if not torch.is_tensor(targets):
        targets = torch.from_numpy(np.array(targets))

    if outputs.dim() == targets.dim() + 1:
        # looks like we have scores/probabilities in our outputs
        # let's convert them to final model predictions
        num_classes = max(outputs.shape[argmax_dim], int(targets.max().detach().item() + 1))
        outputs = torch.argmax(outputs, dim=argmax_dim)
    if num_classes is None:
        # as far as we expect the outputs/targets tensors to be int64
        # we could find number of classes as max available number
        num_classes = max(
            int(outputs.max().detach().item() + 1), int(targets.max().detach().item() + 1),
        )

    if outputs.dim() == 1:
        outputs = outputs.view(-1, 1)
    elif outputs.dim() == 2 and outputs.size(0) == 1:
        # transpose case
        outputs.permute(1, 0)
    else:
        assert outputs.size(1) == 1 and outputs.dim() == 2, (
            "Wrong `outputs` shape, "
            "expected 1D or 2D with size 1 in the second dim "
            "got {}".format(outputs.shape)
        )

    if targets.dim() == 1:
        targets = targets.view(-1, 1)
    elif targets.dim() == 2 and targets.size(0) == 1:
        # transpose case
        targets.permute(1, 0)
    else:
        assert targets.size(1) == 1 and targets.dim() == 2, (
            "Wrong `outputs` shape, " "expected 1D or 2D with size 1 in the second dim"
        )

    return outputs, targets, num_classes


def process_recsys_components(outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    General pre-processing for calculation recsys metrics

    Args:
        outputs (torch.Tensor):
            Tensor with predicted scores
            size: [batch_size, slate_length]
            model outputs, logits
        targets (torch.Tensor):
            Binary tensor with ground truth.
            1 means the item is relevant
            for the user and 0 not relevant
            size: [batch_szie, slate_length]
            ground truth, labels

    Returns:
        targets_sorted_by_outputs (torch.Tensor):
            targets tensor sorted by outputs
    """
    check_consistent_length(outputs, targets)
    outputs_order = torch.argsort(outputs, descending=True, dim=-1)
    targets_sorted_by_outputs = torch.gather(targets, dim=-1, index=outputs_order)
    return targets_sorted_by_outputs


def process_multilabel_components(
    outputs: torch.Tensor, targets: torch.Tensor, weights: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """General preprocessing for multilabel-based metrics.

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
            "wrong `outputs` size " "(should be 1D or 2D with one column per class)"
        )

    if targets.dim() == 1:
        if outputs.shape[1] > 1:
            # multiclass case
            num_classes = outputs.shape[1]
            targets = F.one_hot(targets, num_classes).float()
        else:
            # binary case
            targets = targets.view(-1, 1)
    else:
        assert targets.dim() == 2, (
            "wrong `targets` size " "(should be 1D or 2D with one column per class)"
        )

    if weights is not None:
        assert weights.dim() == 1, "Weights dimension should be 1"
        assert weights.numel() == targets.size(
            0
        ), "Weights dimension 1 should be the same as that of target"
        assert torch.min(weights) >= 0, "Weight should be non-negative only"

    assert torch.equal(targets ** 2, targets), "targets should be binary (0 or 1)"

    return outputs, targets, weights


def get_binary_statistics(
    outputs: Tensor, targets: Tensor, label: int = 1,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """
    Computes the number of true negative, false positive,
    false negative, true positive and support
    for a binary classification problem for a given label.

    Args:
        outputs: estimated targets as predicted by a model
            with shape [bs; ..., 1]
        targets: ground truth (correct) target values
            with shape [bs; ..., 1]
        label: integer, that specifies label of interest for statistics compute

    Returns:
        Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]: stats

    Example:

        >>> y_pred = torch.tensor([[0, 0, 1, 1, 0, 1, 0, 1]])
        >>> y_true = torch.tensor([[0, 1, 0, 1, 0, 0, 1, 1]])
        >>> tn, fp, fn, tp, support = get_binary_statistics(y_pred, y_true)
        tensor(2) tensor(2) tensor(2) tensor(2) tensor(4)

    """
    tn = ((outputs != label) * (targets != label)).to(torch.long).sum()
    fp = ((outputs == label) * (targets != label)).to(torch.long).sum()
    fn = ((outputs != label) * (targets == label)).to(torch.long).sum()
    tp = ((outputs == label) * (targets == label)).to(torch.long).sum()
    support = (targets == label).to(torch.long).sum()
    return tn, fp, fn, tp, support


def get_multiclass_statistics(
    outputs: Tensor, targets: Tensor, argmax_dim: int = -1, num_classes: Optional[int] = None,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """
    Computes the number of true negative, false positive,
    false negative, true positive and support
    for a multiclass classification problem.

    Args:
        outputs: estimated targets as predicted by a model
            with shape [bs; ..., (num_classes or 1)]
        targets: ground truth (correct) target values
            with shape [bs; ..., 1]
        argmax_dim: int, that specifies dimension for argmax transformation
            in case of scores/probabilities in ``outputs``
        num_classes: int, that specifies number of classes if it known

    Returns:
        Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]: stats

    Example:

        >>> y_pred = torch.tensor([1, 2, 3, 0])
        >>> y_true = torch.tensor([1, 3, 4, 0])
        >>> tn, fp, fn, tp, support = get_multiclass_statistics(y_pred, y_true)
        tensor([3., 3., 3., 2., 3.]), tensor([0., 0., 1., 1., 0.]),
        tensor([0., 0., 0., 1., 1.]), tensor([1., 1., 0., 0., 0.]),
        tensor([1., 1., 0., 1., 1.])
    """
    outputs, targets, num_classes = process_multiclass_components(
        outputs=outputs, targets=targets, argmax_dim=argmax_dim, num_classes=num_classes,
    )

    tn = torch.zeros((num_classes,), device=outputs.device)
    fp = torch.zeros((num_classes,), device=outputs.device)
    fn = torch.zeros((num_classes,), device=outputs.device)
    tp = torch.zeros((num_classes,), device=outputs.device)
    support = torch.zeros((num_classes,), device=outputs.device)

    for class_index in range(num_classes):
        (
            tn[class_index],
            fp[class_index],
            fn[class_index],
            tp[class_index],
            support[class_index],
        ) = get_binary_statistics(outputs=outputs, targets=targets, label=class_index)

    return tn, fp, fn, tp, support


def get_multilabel_statistics(
    outputs: Tensor, targets: Tensor,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """
    Computes the number of true negative, false positive,
    false negative, true positive and support
    for a multilabel classification problem.

    Args:
        outputs: estimated targets as predicted by a model
            with shape [bs; ..., (num_classes or 1)]
        targets: ground truth (correct) target values
            with shape [bs; ..., 1]

    Returns:
        Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]: stats

    Example:

        >>> y_pred = torch.tensor([[0, 0, 1, 1], [0, 1, 0, 1]])
        >>> y_true = torch.tensor([[0, 1, 0, 1], [0, 0, 1, 1]])
        >>> tn, fp, fn, tp, support = get_multilabel_statistics(y_pred, y_true)
        tensor([2., 0., 0., 0.]) tensor([0., 1., 1., 0.]),
        tensor([0., 1., 1., 0.]) tensor([0., 0., 0., 2.]),
        tensor([0., 1., 1., 2.])

        >>> y_pred = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        >>> y_true = torch.tensor([0, 1, 2])
        >>> tn, fp, fn, tp, support = get_multilabel_statistics(y_pred, y_true)
        tensor([2., 2., 2.]) tensor([0., 0., 0.])
        tensor([0., 0., 0.]) tensor([1., 1., 1.])
        tensor([1., 1., 1.])

        >>> y_pred = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        >>> y_true = torch.nn.functional.one_hot(torch.tensor([0, 1, 2]))
        >>> tn, fp, fn, tp, support = get_multilabel_statistics(y_pred, y_true)
        tensor([2., 2., 2.]) tensor([0., 0., 0.])
        tensor([0., 0., 0.]) tensor([1., 1., 1.])
        tensor([1., 1., 1.])

    """
    outputs, targets, _ = process_multilabel_components(outputs=outputs, targets=targets)
    assert outputs.shape == targets.shape
    num_classes = outputs.shape[-1]

    tn = torch.zeros((num_classes,), device=outputs.device)
    fp = torch.zeros((num_classes,), device=outputs.device)
    fn = torch.zeros((num_classes,), device=outputs.device)
    tp = torch.zeros((num_classes,), device=outputs.device)
    support = torch.zeros((num_classes,), device=outputs.device)

    for class_index in range(num_classes):
        class_outputs = outputs[..., class_index]
        class_targets = targets[..., class_index]
        (
            tn[class_index],
            fp[class_index],
            fn[class_index],
            tp[class_index],
            support[class_index],
        ) = get_binary_statistics(outputs=class_outputs, targets=class_targets, label=1)

    return tn, fp, fn, tp, support


def get_default_topk_args(num_classes: int) -> Sequence[int]:
    """Calculate list params for ``Accuracy@k`` and ``mAP@k``.

    Args:
        num_classes: number of classes

    Returns:
        iterable: array of accuracy arguments

    Examples:
        >>> get_default_topk_args(num_classes=4)
        [1, 3]

        >>> get_default_topk_args(num_classes=8)
        [1, 3, 5]
    """
    result = [1]

    if num_classes is None:
        return result

    if num_classes > 3:
        result.append(3)
    if num_classes > 5:
        result.append(5)

    return result


def check_consistent_length(*tensors):
    """Check that all arrays have consistent first dimensions.
    Checks whether all objects in arrays have the same shape or length.

    Args:
        tensors : list or tensors of input objects.
            Objects that will be checked for consistent length.

    Raises:
        ValueError: "Inconsistent numbers of samples"

    """
    lengths = [tensor.size(0) * tensor.size(1) for tensor in tensors]
    uniques = np.unique(lengths)
    if len(uniques) > 1:
        raise ValueError("Inconsistent numbers of samples")


__all__ = [
    "check_consistent_length",
    "process_multilabel_components",
    "process_recsys_components",
    "get_binary_statistics",
    "get_multiclass_statistics",
    "get_multilabel_statistics",
    "get_default_topk_args",
]
