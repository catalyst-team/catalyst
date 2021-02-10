from typing import Dict, Iterable, Union

import numpy as np
import pytest

import torch

from catalyst.metrics.accuracy import MultilabelAccuracyMetric


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
            torch.tensor(
                [[0, 1, 1, 0], [1, 0, 1, 0], [0, 1, 0, 0], [0, 1, 0, 0]]
            ),
            0.6,
            {"accuracy": 0.75, "accuracy/std": 0,},
        ),
        (
            torch.tensor([[0.4, 0.9, 0.2], [0.2, 0.8, 0.7], [0.7, 0.9, 0.5],]),
            torch.tensor([[0, 1, 1], [1, 1, 1], [1, 1, 0],]),
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
    Test multilabel accuracy metric with single and multiple thresholds

    Args:
        outputs: tensor of outputs
        targets: tensor of true answers
        thresholds: thresholds for multilabel classification
        true_values: expected metric value
    """
    metric = MultilabelAccuracyMetric(thresholds=thresholds)
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
                torch.tensor([[0, 0, 0],]),
            ],
            [
                torch.tensor([[0, 1, 1], [1, 0, 1], [1, 0, 1]]),
                torch.tensor([[1, 1, 1], [0, 0, 1], [1, 0, 0]]),
                torch.tensor([[0, 0, 0],]),
            ],
            0.7,
            [0.777778, 0.833333, 0.857143],
        ),
        (
            [
                torch.tensor([[0, 1, 0, 1], [1, 0, 1, 0],]),
                torch.tensor([[1, 1, 1, 1], [1, 0, 1, 1],]),
                torch.tensor([[0, 0, 0, 1], [0, 0, 1, 0],]),
            ],
            [
                torch.tensor([[0, 1, 1, 1], [1, 0, 1, 0],]),
                torch.tensor([[1, 1, 1, 1], [1, 0, 0, 0],]),
                torch.tensor([[0, 1, 0, 1], [0, 0, 1, 0],]),
            ],
            0.8,
            [0.875, 0.8125, 0.833333],
        ),
        (
            [
                torch.tensor([[0, 1], [1, 0], [1, 1]]),
                torch.tensor([[1, 1], [0, 0],]),
            ],
            [
                torch.tensor([[0, 1], [0, 0], [0, 0]]),
                torch.tensor([[1, 1], [1, 0],]),
            ],
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
    metric = MultilabelAccuracyMetric(thresholds=thresholds)
    for outputs, targets, true_value in zip(
        outputs_list, targets_list, true_values_list
    ):
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
                torch.tensor([[0, 0, 0],]),
            ],
            [
                torch.tensor([[0, 1, 1], [1, 0, 1], [1, 0, 1]]),
                torch.tensor([[1, 1, 1], [0, 0, 1], [1, 0, 0]]),
                torch.tensor([[0, 0, 0],]),
            ],
            0.7,
            [0, 0.057166, 0.079682],
        ),
        (
            [
                torch.tensor([[0, 1, 0, 1], [1, 0, 1, 0],]),
                torch.tensor([[1, 1, 1, 1], [1, 0, 1, 1],]),
                torch.tensor([[0, 0, 0, 1], [0, 0, 1, 0],]),
            ],
            [
                torch.tensor([[0, 1, 1, 1], [1, 0, 1, 0],]),
                torch.tensor([[1, 1, 1, 1], [1, 0, 0, 0],]),
                torch.tensor([[0, 1, 0, 1], [0, 0, 1, 0],]),
            ],
            0.8,
            [0, 0.06455, 0.0601929],
        ),
        (
            [
                torch.tensor([[0, 1], [1, 0], [1, 1]]),
                torch.tensor([[1, 1], [0, 0],]),
            ],
            [
                torch.tensor([[0, 1], [0, 0], [0, 0]]),
                torch.tensor([[1, 1], [1, 0],]),
            ],
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

    Args:
        outputs_list: list of output tensors
        targets_list: list of true answer tensors
        thresholds: threshold
        true_values_list: true intermediate metric results
    """
    metric = MultilabelAccuracyMetric(thresholds=thresholds)
    for outputs, targets, true_value in zip(
        outputs_list, targets_list, true_values_list
    ):
        metric.update(outputs=outputs, targets=targets)
        _, std = metric.compute()
        assert np.isclose(std, true_value)
