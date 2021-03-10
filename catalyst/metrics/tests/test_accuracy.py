# flake8: noqa
from typing import Dict, Iterable, List, Union

import numpy as np
import pytest
import torch

from catalyst.metrics._accuracy import AccuracyMetric, MultilabelAccuracyMetric


@pytest.mark.parametrize(
    "outputs,targets,num_classes,topk,true_values",
    (
        (
            torch.tensor(
                [
                    [0.2, 0.5, 0.0, 0.3],
                    [0.9, 0.1, 0.0, 0.0],
                    [0.0, 0.1, 0.6, 0.3],
                    [0.0, 0.8, 0.2, 0.0],
                ]
            ),
            torch.tensor([3, 0, 2, 2]),
            4,
            [1, 2],
            {
                "accuracy01": 0.5,
                "accuracy02": 1.0,
                "accuracy": 0.5,
                "accuracy/std": 0.0,
                "accuracy01/std": 0.0,
                "accuracy02/std": 0.0,
            },
        ),
        (
            torch.tensor([[0.1, 0.2, 0.7, 0.0], [0.49, 0.51, 0.0, 0.0], [0.6, 0.3, 0.1, 0.0]]),
            torch.tensor([0, 1, 3]),
            4,
            [1, 3],
            {
                "accuracy01": 0.333333,
                "accuracy03": 0.666667,
                "accuracy": 0.333333,
                "accuracy/std": 0.0,
                "accuracy01/std": 0.0,
                "accuracy03/std": 0.0,
            },
        ),
    ),
)
def test_accuracy(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    topk: List[int],
    true_values: Dict[str, float],
) -> None:
    """
    Test multiclass accuracy with different topk args
    Note that now `accuracy/std` is not std exactly so it can fail if you fix it.

    Args:
        outputs: tensor of outputs
        targets: tensor of targets
        num_classes: number of classes for classification
        topk: list of topk args for accuracy@topk
        true_values: true metrics values
    """
    metric = AccuracyMetric(topk_args=topk)
    metric.update(logits=outputs, targets=targets)
    metrics = metric.compute_key_value()
    for key in true_values.keys():
        assert key in metrics
        assert np.isclose(true_values[key], metrics[key])


@pytest.mark.parametrize(
    "outputs_list,targets_list,num_classes,topk,true_values_list",
    (
        (
            [
                torch.tensor([[0.4, 0.6], [0.7, 0.3]]),
                torch.tensor([[0.8, 0.2], [1.0, 0.0]]),
                torch.tensor([[0.55, 0.45]]),
            ],
            [torch.tensor([0, 0]), torch.tensor([0, 0]), torch.tensor([1])],
            2,
            [1],
            [
                {"accuracy01": 0.5, "accuracy": 0.5, "accuracy/std": 0.0, "accuracy01/std": 0.0},
                {
                    "accuracy01": 0.75,
                    "accuracy": 0.75,
                    "accuracy/std": 0.288675,
                    "accuracy01/std": 0.288675,
                },
                {
                    "accuracy01": 0.6,
                    "accuracy": 0.6,
                    "accuracy/std": 0.41833,
                    "accuracy01/std": 0.41833,
                },
            ],
        ),
        (
            [torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.7, 0.3]]), torch.tensor([[0.0, 0.6, 0.4]])],
            [torch.tensor([2, 2]), torch.tensor([0])],
            3,
            [1, 2],
            [
                {
                    "accuracy01": 0.5,
                    "accuracy02": 1.0,
                    "accuracy": 0.5,
                    "accuracy/std": 0.0,
                    "accuracy01/std": 0.0,
                    "accuracy02/std": 0.0,
                },
                {
                    "accuracy01": 0.333333,
                    "accuracy02": 0.666667,
                    "accuracy": 0.333333,
                    "accuracy/std": 0.288675,
                    "accuracy01/std": 0.288675,
                    "accuracy02/std": 0.57735,
                },
            ],
        ),
    ),
)
def test_accuracy_update(
    outputs_list: List[torch.Tensor],
    targets_list: List[torch.Tensor],
    num_classes: int,
    topk: List[int],
    true_values_list: List[Dict[str, float]],
) -> None:
    """
    This test checks that AccuracyMetric updates its values correctly and return
    correct intermediate results
    Note that now `accuracy/std` is not std exactly so it can fail if you fix it.

    Args:
        outputs_list: list of output tensors
        targets_list: list of target tensors
        num_classes: number od classes for classification
        topk: topk args for computing accuracy@topk
        true_values_list: list of correct metrics intermediate values
    """
    metric = AccuracyMetric(topk_args=topk, num_classes=num_classes)
    for outputs, targets, true_values in zip(outputs_list, targets_list, true_values_list):
        metric.update(logits=outputs, targets=targets)
        intermediate_metric_values = metric.compute_key_value()
        for key in true_values.keys():
            assert key in intermediate_metric_values
            assert np.isclose(true_values[key], intermediate_metric_values[key])


@pytest.mark.parametrize(
    "outputs,targets,thresholds,true_values",
    (
        (
            torch.tensor(
                [
                    [0.1, 0.9, 0.0, 0.8],
                    [0.96, 0.01, 0.85, 0.2],
                    [0.98, 0.4, 0.2, 0.1],
                    [0.1, 0.89, 0.2, 0.0],
                ]
            ),
            torch.tensor([[0, 1, 1, 0], [1, 0, 1, 0], [0, 1, 0, 0], [0, 1, 0, 0]]),
            0.6,
            {"accuracy": 0.75, "accuracy/std": 0},
        ),
        (
            torch.tensor([[0.4, 0.9, 0.2], [0.2, 0.8, 0.7], [0.7, 0.9, 0.5]]),
            torch.tensor([[0, 1, 1], [1, 1, 1], [1, 1, 0]]),
            torch.tensor([0.5, 0.7, 0.6]),
            {"accuracy": 0.777778, "accuracy/std": 0},
        ),
    ),
)
def test_multilabel_accuracy(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    thresholds: Union[float, torch.Tensor],
    true_values: Dict[str, float],
) -> None:
    """
    Test multilabel accuracy metric with single and multiple thresholds.
    Note that now `accuracy/std` is not std exactly so it can fail if you fix it.

    Args:
        outputs: tensor of outputs
        targets: tensor of true answers
        thresholds: thresholds for multilabel classification
        true_values: expected metric value
    """
    metric = MultilabelAccuracyMetric(threshold=thresholds)
    metric.update(outputs=outputs, targets=targets)
    values = metric.compute_key_value()
    for key in true_values.keys():
        assert key in true_values
        assert np.isclose(true_values[key], values[key])


@pytest.mark.parametrize(
    "outputs_list,targets_list,thresholds,true_values_list",
    (
        (
            [
                torch.tensor([[0, 1, 0], [1, 0, 1], [1, 1, 1]]),
                torch.tensor([[1, 1, 1], [1, 0, 1], [1, 0, 0]]),
                torch.tensor([[0, 0, 0]]),
            ],
            [
                torch.tensor([[0, 1, 1], [1, 0, 1], [1, 0, 1]]),
                torch.tensor([[1, 1, 1], [0, 0, 1], [1, 0, 0]]),
                torch.tensor([[0, 0, 0]]),
            ],
            0.7,
            [0.777778, 0.833333, 0.857143],
        ),
        (
            [
                torch.tensor([[0, 1, 0, 1], [1, 0, 1, 0]]),
                torch.tensor([[1, 1, 1, 1], [1, 0, 1, 1]]),
                torch.tensor([[0, 0, 0, 1], [0, 0, 1, 0]]),
            ],
            [
                torch.tensor([[0, 1, 1, 1], [1, 0, 1, 0]]),
                torch.tensor([[1, 1, 1, 1], [1, 0, 0, 0]]),
                torch.tensor([[0, 1, 0, 1], [0, 0, 1, 0]]),
            ],
            0.8,
            [0.875, 0.8125, 0.833333],
        ),
        (
            [torch.tensor([[0, 1], [1, 0], [1, 1]]), torch.tensor([[1, 1], [0, 0]])],
            [torch.tensor([[0, 1], [0, 0], [0, 0]]), torch.tensor([[1, 1], [1, 0]])],
            torch.tensor([0.5, 0.6]),
            [0.5, 0.6],
        ),
    ),
)
def test_multilabel_accuracy_mean(
    outputs_list: Iterable[torch.Tensor],
    targets_list: Iterable[torch.Tensor],
    thresholds: Union[float, torch.Tensor],
    true_values_list: Iterable[float],
) -> None:
    """
    This test checks that all the intermediate metrics values are correct during accumulation.

    Args:
        outputs_list: list of output tensors
        targets_list: list of true answer tensors
        thresholds: threshold
        true_values_list: true intermediate metric results
    """
    metric = MultilabelAccuracyMetric(threshold=thresholds)
    for outputs, targets, true_value in zip(outputs_list, targets_list, true_values_list):
        metric.update(outputs=outputs, targets=targets)
        mean, _ = metric.compute()
        assert np.isclose(mean, true_value)


@pytest.mark.parametrize(
    "outputs_list,targets_list,thresholds,true_values_list",
    (
        (
            [
                torch.tensor([[0, 1, 0], [1, 0, 1], [1, 1, 1]]),
                torch.tensor([[1, 1, 1], [1, 0, 1], [1, 0, 0]]),
                torch.tensor([[0, 0, 0]]),
            ],
            [
                torch.tensor([[0, 1, 1], [1, 0, 1], [1, 0, 1]]),
                torch.tensor([[1, 1, 1], [0, 0, 1], [1, 0, 0]]),
                torch.tensor([[0, 0, 0]]),
            ],
            0.7,
            [0, 0.057166, 0.079682],
        ),
        (
            [
                torch.tensor([[0, 1, 0, 1], [1, 0, 1, 0]]),
                torch.tensor([[1, 1, 1, 1], [1, 0, 1, 1]]),
                torch.tensor([[0, 0, 0, 1], [0, 0, 1, 0]]),
            ],
            [
                torch.tensor([[0, 1, 1, 1], [1, 0, 1, 0]]),
                torch.tensor([[1, 1, 1, 1], [1, 0, 0, 0]]),
                torch.tensor([[0, 1, 0, 1], [0, 0, 1, 0]]),
            ],
            0.8,
            [0, 0.06455, 0.0601929],
        ),
        (
            [torch.tensor([[0, 1], [1, 0], [1, 1]]), torch.tensor([[1, 1], [0, 0]])],
            [torch.tensor([[0, 1], [0, 0], [0, 0]]), torch.tensor([[1, 1], [1, 0]])],
            torch.tensor([0.5, 0.6]),
            [0, 0.129099],
        ),
    ),
)
def test_multilabel_accuracy_std(
    outputs_list: Iterable[torch.Tensor],
    targets_list: Iterable[torch.Tensor],
    thresholds: Union[float, torch.Tensor],
    true_values_list: Iterable[float],
) -> None:
    """
    This test checks that all the intermediate metrics values are correct during accumulation.
    Note that now `accuracy/std` is not std exactly so it can fail if you fix it.

    Args:
        outputs_list: list of output tensors
        targets_list: list of true answer tensors
        thresholds: threshold
        true_values_list: true intermediate metric results
    """
    metric = MultilabelAccuracyMetric(threshold=thresholds)
    for outputs, targets, true_value in zip(outputs_list, targets_list, true_values_list):
        metric.update(outputs=outputs, targets=targets)
        _, std = metric.compute()
        assert np.isclose(std, true_value)
