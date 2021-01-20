from typing import Callable, Dict, Optional, Sequence, Tuple
from functools import partial
import logging

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from catalyst.utils.torch import get_activation_fn

# @TODO:
# after full classification metrics re-implementation, make a reference to
# https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/metrics
# as a baseline

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
            Tensor weith predicted score
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
    false negative, true negative and support
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
    false negative, true negative and support
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
    false negative, true negative and support
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


def wrap_metric_fn_with_activation(
    metric_fn: Callable, activation: str = None,
):
    """Wraps model outputs for ``metric_fn` with specified ``activation``.

    Args:
        metric_fn: metric function to compute
        activation: activation name to use

    Returns:
        wrapped metric function with wrapped model outputs

    .. note::
        Works only with ``metric_fn`` like
        ``metric_fn(outputs, targets, *args, **kwargs)``.
    """
    activation_fn = get_activation_fn(activation)

    def wrapped_metric_fn(outputs: torch.Tensor, targets: torch.Tensor, *args, **kwargs):
        outputs = activation_fn(outputs)
        output = metric_fn(outputs, targets, *args, **kwargs)
        return output

    return wrapped_metric_fn


def wrap_class_metric2dict(
    metric_fn: Callable, per_class: bool = False, class_args: Sequence[str] = None,
) -> Callable:
    """# noqa: D202
    Logging wrapper for metrics with torch.Tensor output
    and [num_classes] shape.
    Computes the metric and sync each element from the output Tensor
    with passed `class` argument.

    Args:
        metric_fn: metric function to compute
        per_class: boolean flag to log per class metrics,
            or use mean/macro statistics otherwise
        class_args: class names for logging,
            default: None - class indexes will be used.

    Returns:
        wrapped metric function with List[Dict] output
    """
    if per_class is False and class_args is not None:
        logger.warning(
            "``per_class`` is disabled, but ``class_args`` are not None"
            "check the experiment conditions."
        )

    if per_class:

        def class_metric_with_dict_output(*args, **kwargs):
            output = metric_fn(*args, **kwargs)
            num_classes = len(output)
            output_class_args = class_args or [f"/class_{i:02}" for i in range(num_classes)]
            mean_stats = torch.mean(output).item()
            output = {key: value.item() for key, value in zip(output_class_args, output)}
            output[""] = mean_stats
            output["/mean"] = mean_stats
            return output

    else:

        def class_metric_with_dict_output(*args, **kwargs):
            output = metric_fn(*args, **kwargs)
            mean_stats = torch.mean(output).item()
            output = {"": mean_stats}
            return output

    return class_metric_with_dict_output


def wrap_topk_metric2dict(
    metric_fn: Callable, topk_args: Sequence[int]
) -> Callable:  # noqa: DAR401
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
    """
    metric_fn = partial(metric_fn, topk=topk_args)

    def topk_metric_with_dict_output(*args, **kwargs):
        output: Sequence = metric_fn(*args, **kwargs)

        if isinstance(output[0], (int, float, torch.Tensor)):
            output = {
                f"{topk_key:02}": metric_value for topk_key, metric_value in zip(topk_args, output)
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
    "check_consistent_length",
    "process_multilabel_components",
    "process_recsys_components",
    "get_binary_statistics",
    "get_multiclass_statistics",
    "get_multilabel_statistics",
    "get_default_topk_args",
    "wrap_metric_fn_with_activation",
    "wrap_topk_metric2dict",
    "wrap_class_metric2dict",
]
