# flake8: noqa
from typing import Dict, Iterable

import pytest
import torch

from catalyst.metrics import (
    BinaryPrecisionRecallF1Metric,
    MulticlassPrecisionRecallF1SupportMetric,
    MultilabelPrecisionRecallF1SupportMetric,
)

EPS = 1e-5


@pytest.mark.parametrize(
    "outputs,targets,num_classes,zero_division,true_values",
    (
        (
            torch.tensor([1, 0, 3, 2, 2, 2]),
            torch.tensor([0, 0, 2, 4, 3, 1]),
            5,
            1,
            {
                "precision/_macro": 0.4,
                "precision/_micro": 0.166667,
                "precision/_weighted": 0.5,
                "precision/class_00": 1.0,
                "precision/class_01": 0.0,
                "precision/class_02": 0.0,
                "precision/class_03": 0.0,
                "precision/class_04": 1.0,
                "recall/_macro": 0.1,
                "recall/_micro": 0.166667,
                "recall/_weighted": 0.166667,
                "recall/class_00": 0.5,
                "recall/class_01": 0.0,
                "recall/class_02": 0.0,
                "recall/class_03": 0.0,
                "recall/class_04": 0.0,
                "f1/_macro": 0.133333,
                "f1/_micro": 0.166667,
                "f1/_weighted": 0.222222,
                "f1/class_00": 0.666667,
                "f1/class_01": 0.0,
                "f1/class_02": 0.0,
                "f1/class_03": 0.0,
                "f1/class_04": 0.0,
                "support/class_00": 2,
                "support/class_01": 1,
                "support/class_02": 1,
                "support/class_03": 1,
                "support/class_04": 1,
            },
        ),
        (
            torch.tensor([2, 2, 2, 3, 3, 4]),
            torch.tensor([4, 4, 2, 1, 3, 1]),
            5,
            0,
            {
                "precision/_macro": 0.166667,
                "precision/_micro": 0.333333,
                "precision/_weighted": 0.138889,
                "precision/class_00": 0.0,
                "precision/class_01": 0.0,
                "precision/class_02": 0.333333,
                "precision/class_03": 0.5,
                "precision/class_04": 0.0,
                "recall/_macro": 0.4,
                "recall/_micro": 0.333333,
                "recall/_weighted": 0.333333,
                "recall/class_00": 0.0,
                "recall/class_01": 0.0,
                "recall/class_02": 1.0,
                "recall/class_03": 1.0,
                "recall/class_04": 0.0,
                "f1/_macro": 0.233333,
                "f1/_micro": 0.333333,
                "f1/_weighted": 0.194444,
                "f1/class_00": 0.0,
                "f1/class_01": 0.0,
                "f1/class_02": 0.5,
                "f1/class_03": 0.666667,
                "f1/class_04": 0.0,
                "support/class_00": 0,
                "support/class_01": 2,
                "support/class_02": 1,
                "support/class_03": 1,
                "support/class_04": 2,
            },
        ),
        (
            torch.tensor([2, 2, 2, 3, 3, 4]),
            torch.tensor([4, 4, 2, 1, 3, 1]),
            5,
            1,
            {
                "precision/_macro": 0.566667,
                "precision/_micro": 0.333333,
                "precision/_weighted": 0.472222,
                "precision/class_00": 1.0,
                "precision/class_01": 1.0,
                "precision/class_02": 0.333333,
                "precision/class_03": 0.5,
                "precision/class_04": 0.0,
                "recall/_macro": 0.6,
                "recall/_micro": 0.333333,
                "recall/_weighted": 0.333333,
                "recall/class_00": 1.0,
                "recall/class_01": 0.0,
                "recall/class_02": 1.0,
                "recall/class_03": 1.0,
                "recall/class_04": 0.0,
                "f1/_macro": 0.433333,
                "f1/_micro": 0.333333,
                "f1/_weighted": 0.194444,
                "f1/class_00": 1.0,
                "f1/class_01": 0.0,
                "f1/class_02": 0.5,
                "f1/class_03": 0.666667,
                "f1/class_04": 0.0,
                "support/class_00": 0,
                "support/class_01": 2,
                "support/class_02": 1,
                "support/class_03": 1,
                "support/class_04": 2,
            },
        ),
        (
            torch.tensor([5, 1, 4, 0, 4, 6, 2, 2, 0, 5]),
            torch.tensor([1, 2, 1, 1, 2, 5, 2, 0, 6, 6]),
            7,
            1,
            {
                "precision/_macro": 0.214286,
                "precision/_micro": 0.1,
                "precision/_weighted": 0.15,
                "precision/class_00": 0.0,
                "precision/class_01": 0.0,
                "precision/class_02": 0.5,
                "precision/class_03": 1.0,
                "precision/class_04": 0.0,
                "precision/class_05": 0.0,
                "precision/class_06": 0.0,
                "recall/_macro": 0.333333,
                "recall/_micro": 0.1,
                "recall/_weighted": 0.1,
                "recall/class_00": 0.0,
                "recall/class_01": 0.0,
                "recall/class_02": 0.333333,
                "recall/class_03": 1.0,
                "recall/class_04": 1.0,
                "recall/class_05": 0.0,
                "recall/class_06": 0.0,
                "f1/_macro": 0.2,
                "f1/_micro": 0.1,
                "f1/_weighted": 0.12,
                "f1/class_00": 0.0,
                "f1/class_01": 0.0,
                "f1/class_02": 0.4,
                "f1/class_03": 1.0,
                "f1/class_04": 0.0,
                "f1/class_05": 0.0,
                "f1/class_06": 0.0,
                "support/class_00": 1,
                "support/class_01": 3,
                "support/class_02": 3,
                "support/class_03": 0,
                "support/class_04": 0,
                "support/class_05": 1,
                "support/class_06": 2,
            },
        ),
        (
            torch.tensor([2, 2, 1, 1, 2, 2, 1, 2, 2, 0]),
            torch.tensor([1, 2, 0, 2, 2, 1, 1, 2, 0, 2]),
            3,
            0,
            {
                "precision/_macro": 0.277778,
                "precision/_micro": 0.4,
                "precision/_weighted": 0.35,
                "precision/class_00": 0.0,
                "precision/class_01": 0.333333,
                "precision/class_02": 0.5,
                "recall/_macro": 0.311111,
                "recall/_micro": 0.4,
                "recall/_weighted": 0.4,
                "recall/class_00": 0.0,
                "recall/class_01": 0.333333,
                "recall/class_02": 0.6,
                "f1/_macro": 0.292929,
                "f1/_micro": 0.4,
                "f1/_weighted": 0.372727,
                "f1/class_00": 0.0,
                "f1/class_01": 0.333333,
                "f1/class_02": 0.545455,
                "support/class_00": 2,
                "support/class_01": 3,
                "support/class_02": 5,
            },
        ),
        (
            torch.tensor([2, 0, 0, 0]),
            torch.tensor([2, 2, 0, 2]),
            3,
            0,
            {
                "precision/_macro": 0.444444,
                "precision/_micro": 0.5,
                "precision/_weighted": 0.833333,
                "precision/class_00": 0.333333,
                "precision/class_01": 0.0,
                "precision/class_02": 1.0,
                "recall/_macro": 0.444444,
                "recall/_micro": 0.5,
                "recall/_weighted": 0.5,
                "recall/class_00": 1.0,
                "recall/class_01": 0.0,
                "recall/class_02": 0.333333,
                "f1/_macro": 0.333333,
                "f1/_micro": 0.5,
                "f1/_weighted": 0.5,
                "f1/class_00": 0.5,
                "f1/class_01": 0.0,
                "f1/class_02": 0.5,
                "support/class_00": 1,
                "support/class_01": 0,
                "support/class_02": 3,
            },
        ),
    ),
)
def test_multiclass_metrics(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    zero_division: int,
    true_values: Dict[str, float],
) -> None:
    """
    Test multiclass metric
    Args:
        outputs: tensor of predictions
        targets: tensor of targets
        zero_division: zero division policy flag
        true_values: true values of metrics
    """
    metric = MulticlassPrecisionRecallF1SupportMetric(
        num_classes=num_classes, zero_division=zero_division
    )
    metric.update(outputs=outputs, targets=targets)
    metrics = metric.compute_key_value()
    for key in true_values:
        assert key in metrics
        assert abs(metrics[key] - true_values[key]) < EPS


@pytest.mark.parametrize(
    "outputs,targets,num_classes,zero_division,true_values",
    (
        (
            torch.tensor([[0, 1, 0], [1, 1, 0], [0, 0, 1], [0, 0, 0], [0, 1, 1]]),
            torch.tensor([[0, 1, 1], [1, 1, 1], [0, 0, 0], [0, 0, 1], [0, 1, 1]]),
            3,
            0,
            {
                "precision/_macro": 0.833333,
                "precision/_micro": 0.833333,
                "precision/_weighted": 0.75,
                "precision/class_00": 1.0,
                "precision/class_01": 1.0,
                "precision/class_02": 0.5,
                "recall/_macro": 0.75,
                "recall/_micro": 0.625,
                "recall/_weighted": 0.625,
                "recall/class_00": 1.0,
                "recall/class_01": 1.0,
                "recall/class_02": 0.25,
                "f1/_macro": 0.777778,
                "f1/_micro": 0.714286,
                "f1/_weighted": 0.666667,
                "f1/class_00": 1.0,
                "f1/class_01": 1.0,
                "f1/class_02": 0.333333,
                "support/class_00": 1,
                "support/class_01": 3,
                "support/class_02": 4,
            },
        ),
        (
            torch.tensor([[0, 1, 0, 0], [1, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0], [0, 1, 1, 0]]),
            torch.tensor([[0, 1, 1, 1], [1, 1, 1, 0], [0, 0, 0, 0], [0, 0, 1, 1], [0, 1, 1, 0]]),
            4,
            0,
            {
                "precision/_macro": 0.625,
                "precision/_micro": 0.833333,
                "precision/_weighted": 0.6,
                "precision/class_00": 1.0,
                "precision/class_01": 1.0,
                "precision/class_02": 0.5,
                "precision/class_03": 0.0,
                "recall/_macro": 0.5625,
                "recall/_micro": 0.5,
                "recall/_weighted": 0.5,
                "recall/class_00": 1.0,
                "recall/class_01": 1.0,
                "recall/class_02": 0.25,
                "recall/class_03": 0.0,
                "f1/_macro": 0.583333,
                "f1/_micro": 0.625,
                "f1/_weighted": 0.533333,
                "f1/class_00": 1.0,
                "f1/class_01": 1.0,
                "f1/class_02": 0.333333,
                "f1/class_03": 0.0,
                "support/class_00": 1,
                "support/class_01": 3,
                "support/class_02": 4,
                "support/class_03": 2,
            },
        ),
        (
            torch.tensor([[0, 1, 0, 0], [1, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0], [0, 1, 1, 0]]),
            torch.tensor([[0, 1, 1, 1], [1, 1, 1, 0], [0, 0, 0, 0], [0, 0, 1, 1], [0, 1, 1, 0]]),
            4,
            1,
            {
                "precision/_macro": 0.875,
                "precision/_micro": 0.833333,
                "precision/_weighted": 0.8,
                "precision/class_00": 1.0,
                "precision/class_01": 1.0,
                "precision/class_02": 0.5,
                "precision/class_03": 1.0,
                "recall/_macro": 0.5625,
                "recall/_micro": 0.5,
                "recall/_weighted": 0.5,
                "recall/class_00": 1.0,
                "recall/class_01": 1.0,
                "recall/class_02": 0.25,
                "recall/class_03": 0.0,
                "f1/_macro": 0.583333,
                "f1/_micro": 0.625,
                "f1/_weighted": 0.533333,
                "f1/class_00": 1.0,
                "f1/class_01": 1.0,
                "f1/class_02": 0.333333,
                "f1/class_03": 0.0,
                "support/class_00": 1,
                "support/class_01": 3,
                "support/class_02": 4,
                "support/class_03": 2,
            },
        ),
        (
            torch.tensor(
                [
                    [0, 1, 0, 0, 1],
                    [1, 1, 0, 0, 1],
                    [0, 0, 1, 0, 1],
                    [0, 0, 0, 0, 0],
                    [0, 1, 1, 0, 0],
                    [0, 1, 0, 0, 1],
                    [1, 1, 0, 0, 1],
                    [0, 0, 1, 0, 1],
                    [0, 1, 1, 0, 0],
                ]
            ),
            torch.tensor(
                [
                    [0, 1, 1, 1, 0],
                    [1, 1, 1, 0, 1],
                    [0, 0, 0, 0, 0],
                    [0, 0, 1, 1, 0],
                    [0, 1, 1, 0, 0],
                    [0, 1, 1, 1, 0],
                    [1, 1, 1, 0, 1],
                    [0, 0, 0, 0, 0],
                    [0, 0, 1, 1, 0],
                ]
            ),
            5,
            0,
            {
                "precision/_macro": 0.533333,
                "precision/_micro": 0.611111,
                "precision/_weighted": 0.516667,
                "precision/class_00": 1.0,
                "precision/class_01": 0.833333,
                "precision/class_02": 0.5,
                "precision/class_03": 0.0,
                "precision/class_04": 0.333333,
                "recall/_macro": 0.657143,
                "recall/_micro": 0.55,
                "recall/_weighted": 0.55,
                "recall/class_00": 1.0,
                "recall/class_01": 1.0,
                "recall/class_02": 0.285714,
                "recall/class_03": 0.0,
                "recall/class_04": 1.0,
                "f1/_macro": 0.554545,
                "f1/_micro": 0.578947,
                "f1/_weighted": 0.504545,
                "f1/class_00": 1.0,
                "f1/class_01": 0.909091,
                "f1/class_02": 0.363636,
                "f1/class_03": 0.0,
                "f1/class_04": 0.5,
                "support/class_00": 2,
                "support/class_01": 5,
                "support/class_02": 7,
                "support/class_03": 4,
                "support/class_04": 2,
            },
        ),
        (
            torch.tensor([[0, 1], [1, 0], [0, 1], [0, 0], [1, 1]]),
            torch.tensor([[1, 1], [1, 1], [0, 0], [0, 1], [1, 1]]),
            2,
            0,
            {
                "precision/_macro": 0.833333,
                "precision/_micro": 0.8,
                "precision/_weighted": 0.809524,
                "precision/class_00": 1.0,
                "precision/class_01": 0.666667,
                "recall/_macro": 0.583333,
                "recall/_micro": 0.571429,
                "recall/_weighted": 0.571429,
                "recall/class_00": 0.666667,
                "recall/class_01": 0.5,
                "f1/_macro": 0.685714,
                "f1/_micro": 0.666667,
                "f1/_weighted": 0.669388,
                "f1/class_00": 0.8,
                "f1/class_01": 0.571429,
                "support/class_00": 3,
                "support/class_01": 4,
            },
        ),
    ),
)
def test_multilabel_metrics(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    zero_division: int,
    true_values: Dict[str, float],
) -> None:
    """
    Test multilabel metric
    Args:
        outputs: tensor of predictions
        targets: tensor of targets
        zero_division: zero division policy flag
        true_values: true values of metrics
    """
    metric = MultilabelPrecisionRecallF1SupportMetric(
        num_classes=num_classes, zero_division=zero_division
    )
    metric.update(outputs=outputs, targets=targets)
    metrics = metric.compute_key_value()
    for key in true_values:
        assert key in metrics
        assert abs(metrics[key] - true_values[key]) < EPS


@pytest.mark.parametrize(
    "outputs,targets,zero_division,true_values",
    (
        (
            torch.tensor([0, 1, 0, 1, 0, 1, 0, 1]),
            torch.tensor([1, 0, 1, 1, 0, 1, 0, 1]),
            0,
            {"precision": 0.75, "recall": 0.6, "f1": 0.666667},
        ),
        (
            torch.tensor([0, 1, 0, 1, 0, 1, 0, 1]),
            torch.tensor([0, 0, 0, 0, 0, 0, 0, 1]),
            1,
            {"precision": 0.25, "recall": 1, "f1": 0.4},
        ),
        (
            torch.tensor([1, 1, 1, 0]),
            torch.tensor([0, 0, 0, 0]),
            0,
            {"precision": 0, "recall": 0, "f1": 0},
        ),
        (
            torch.tensor([1, 1, 1, 0]),
            torch.tensor([0, 0, 0, 0]),
            1,
            {"precision": 0, "recall": 1, "f1": 0},
        ),
    ),
)
def test_binary_metrics(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    zero_division: int,
    true_values: Dict[str, float],
) -> None:
    """
    Test binary metric
    Args:
        outputs: tensor of predictions
        targets: tensor of targets
        zero_division: zero division policy flag
        true_values: true values of metrics
    """
    metric = BinaryPrecisionRecallF1Metric(zero_division=zero_division)
    metric.update(outputs=outputs, targets=targets)
    metrics = metric.compute_key_value()
    for key in true_values:
        assert key in metrics
        assert abs(metrics[key] - true_values[key]) < EPS


@pytest.mark.parametrize(
    "outputs_list,targets_list,num_classes,zero_division,true_values",
    (
        (
            [torch.tensor([1, 2, 3]), torch.tensor([0, 3, 4]), torch.tensor([4, 5])],
            [torch.tensor([1, 2, 4]), torch.tensor([0, 3, 4]), torch.tensor([5, 5])],
            6,
            0,
            {
                "precision/_macro": 0.833333,
                "precision/_micro": 0.75,
                "precision/_weighted": 0.8125,
                "precision/class_00": 1.0,
                "precision/class_01": 1.0,
                "precision/class_02": 1.0,
                "precision/class_03": 0.5,
                "precision/class_04": 0.5,
                "precision/class_05": 1.0,
                "recall/_macro": 0.833333,
                "recall/_micro": 0.75,
                "recall/_weighted": 0.75,
                "recall/class_00": 1.0,
                "recall/class_01": 1.0,
                "recall/class_02": 1.0,
                "recall/class_03": 1.0,
                "recall/class_04": 0.5,
                "recall/class_05": 0.5,
                "f1/_macro": 0.805556,
                "f1/_micro": 0.75,
                "f1/_weighted": 0.75,
                "f1/class_00": 1.0,
                "f1/class_01": 1.0,
                "f1/class_02": 1.0,
                "f1/class_03": 0.666667,
                "f1/class_04": 0.5,
                "f1/class_05": 0.666667,
                "support/class_00": 1,
                "support/class_01": 1,
                "support/class_02": 1,
                "support/class_03": 1,
                "support/class_04": 2,
                "support/class_05": 2,
            },
        ),
    ),
)
def test_update(
    outputs_list: Iterable[torch.Tensor],
    targets_list: Iterable[torch.Tensor],
    num_classes: int,
    zero_division: int,
    true_values: Dict[str, float],
) -> None:
    """
    Test if metric works correctly while updating.
    Args:
        outputs_list: list of tensors of predictions
        targets_list: list of tensors of targets
        num_classes: number of classes to score
        zero_division: zero division policy flag
        true_values: true values of metrics
    """
    metric = MulticlassPrecisionRecallF1SupportMetric(
        num_classes=num_classes, zero_division=zero_division
    )
    for outputs, targets in zip(outputs_list, targets_list):
        metric.update(outputs=outputs, targets=targets)
    metrics = metric.compute_key_value()
    for key in true_values:
        assert key in metrics
        assert abs(metrics[key] - true_values[key]) < EPS


@pytest.mark.parametrize(
    "outputs_list,targets_list,num_classes,zero_division,update_true_values,compute_true_value",
    (
        (
            [torch.tensor([0, 1, 2]), torch.tensor([2, 3]), torch.tensor([0, 1, 3])],
            [torch.tensor([0, 1, 1]), torch.tensor([2, 3]), torch.tensor([0, 1, 2])],
            4,
            0,
            [
                {
                    "precision/_micro": 0.666667,
                    "recall/_micro": 0.666667,
                    "f1/_micro": 0.666667,
                    "precision/_macro": 0.5,
                    "recall/_macro": 0.375,
                    "f1/_macro": 0.416667,
                    "precision/_weighted": 1.0,
                    "recall/_weighted": 0.666667,
                    "f1/_weighted": 0.777778,
                    "precision/class_00": 1.0,
                    "precision/class_01": 1.0,
                    "precision/class_02": 0.0,
                    "precision/class_03": 0.0,
                    "recall/class_00": 1.0,
                    "recall/class_01": 0.5,
                    "recall/class_02": 0.0,
                    "recall/class_03": 0.0,
                    "f1/class_00": 1.0,
                    "f1/class_01": 0.666667,
                    "f1/class_02": 0.0,
                    "f1/class_03": 0.0,
                    "support/class_00": 1,
                    "support/class_01": 2,
                    "support/class_02": 0,
                    "support/class_03": 0,
                },
                {
                    "precision/_micro": 1.0,
                    "recall/_micro": 1.0,
                    "f1/_micro": 1.0,
                    "precision/_macro": 0.5,
                    "recall/_macro": 0.5,
                    "f1/_macro": 0.5,
                    "precision/_weighted": 1.0,
                    "recall/_weighted": 1.0,
                    "f1/_weighted": 1.0,
                    "precision/class_00": 0.0,
                    "precision/class_01": 0.0,
                    "precision/class_02": 1.0,
                    "precision/class_03": 1.0,
                    "recall/class_00": 0.0,
                    "recall/class_01": 0.0,
                    "recall/class_02": 1.0,
                    "recall/class_03": 1.0,
                    "f1/class_00": 0.0,
                    "f1/class_01": 0.0,
                    "f1/class_02": 1.0,
                    "f1/class_03": 1.0,
                    "support/class_00": 0,
                    "support/class_01": 0,
                    "support/class_02": 1,
                    "support/class_03": 1,
                },
                {
                    "precision/_micro": 0.666667,
                    "recall/_micro": 0.666667,
                    "f1/_micro": 0.666667,
                    "precision/_macro": 0.5,
                    "recall/_macro": 0.5,
                    "f1/_macro": 0.5,
                    "precision/_weighted": 0.666667,
                    "recall/_weighted": 0.666667,
                    "f1/_weighted": 0.666667,
                    "precision/class_00": 1.0,
                    "precision/class_01": 1.0,
                    "precision/class_02": 0.0,
                    "precision/class_03": 0.0,
                    "recall/class_00": 1.0,
                    "recall/class_01": 1.0,
                    "recall/class_02": 0.0,
                    "recall/class_03": 0.0,
                    "f1/class_00": 1.0,
                    "f1/class_01": 1.0,
                    "f1/class_02": 0.0,
                    "f1/class_03": 0.0,
                    "support/class_00": 1,
                    "support/class_01": 1,
                    "support/class_02": 1,
                    "support/class_03": 0,
                },
            ],
            {
                "precision/_micro": 0.75,
                "recall/_micro": 0.75,
                "f1/_micro": 0.75,
                "precision/_macro": 0.75,
                "recall/_macro": 0.791667,
                "f1/_macro": 0.741667,
                "precision/_weighted": 0.8125,
                "recall/_weighted": 0.75,
                "f1/_weighted": 0.758333,
                "precision/class_00": 1.0,
                "precision/class_01": 1.0,
                "precision/class_02": 0.5,
                "precision/class_03": 0.5,
                "recall/class_00": 1.0,
                "recall/class_01": 0.666667,
                "recall/class_02": 0.5,
                "recall/class_03": 1.0,
                "f1/class_00": 1.0,
                "f1/class_01": 0.8,
                "f1/class_02": 0.5,
                "f1/class_03": 0.666667,
                "support/class_00": 2,
                "support/class_01": 3,
                "support/class_02": 2,
                "support/class_03": 1,
            },
        ),
        (
            [torch.tensor([0, 1, 2, 4]), torch.tensor([2, 3, 3, 2]), torch.tensor([0, 1, 3, 4]),],
            [torch.tensor([0, 1, 1, 4]), torch.tensor([2, 3, 3, 4]), torch.tensor([0, 1, 2, 4]),],
            5,
            1,
            [
                {
                    "precision/_micro": 0.75,
                    "recall/_micro": 0.75,
                    "f1/_micro": 0.75,
                    "precision/_macro": 0.8,
                    "recall/_macro": 0.9,
                    "f1/_macro": 0.733333,
                    "precision/_weighted": 1.0,
                    "recall/_weighted": 0.75,
                    "f1/_weighted": 0.833333,
                    "precision/class_00": 1.0,
                    "precision/class_01": 1.0,
                    "precision/class_02": 0.0,
                    "precision/class_03": 1.0,
                    "precision/class_04": 1.0,
                    "recall/class_00": 1.0,
                    "recall/class_01": 0.5,
                    "recall/class_02": 1.0,
                    "recall/class_03": 1.0,
                    "recall/class_04": 1.0,
                    "f1/class_00": 1.0,
                    "f1/class_01": 0.666667,
                    "f1/class_02": 0.0,
                    "f1/class_03": 1.0,
                    "f1/class_04": 1.0,
                    "support/class_00": 1,
                    "support/class_01": 2,
                    "support/class_02": 0,
                    "support/class_03": 0,
                    "support/class_04": 1,
                },
                {
                    "precision/_micro": 0.75,
                    "recall/_micro": 0.75,
                    "f1/_micro": 0.75,
                    "precision/_macro": 0.9,
                    "recall/_macro": 0.8,
                    "f1/_macro": 0.733333,
                    "precision/_weighted": 0.875,
                    "recall/_weighted": 0.75,
                    "f1/_weighted": 0.666667,
                    "precision/class_00": 1.0,
                    "precision/class_01": 1.0,
                    "precision/class_02": 0.5,
                    "precision/class_03": 1.0,
                    "precision/class_04": 1.0,
                    "recall/class_00": 1.0,
                    "recall/class_01": 1.0,
                    "recall/class_02": 1.0,
                    "recall/class_03": 1.0,
                    "recall/class_04": 0.0,
                    "f1/class_00": 1.0,
                    "f1/class_01": 1.0,
                    "f1/class_02": 0.666667,
                    "f1/class_03": 1.0,
                    "f1/class_04": 0.0,
                    "support/class_00": 0,
                    "support/class_01": 0,
                    "support/class_02": 1,
                    "support/class_03": 2,
                    "support/class_04": 1,
                },
                {
                    "precision/_micro": 0.75,
                    "recall/_micro": 0.75,
                    "f1/_micro": 0.75,
                    "precision/_macro": 0.8,
                    "recall/_macro": 0.8,
                    "f1/_macro": 0.6,
                    "precision/_weighted": 1.0,
                    "recall/_weighted": 0.75,
                    "f1/_weighted": 0.75,
                    "precision/class_00": 1.0,
                    "precision/class_01": 1.0,
                    "precision/class_02": 1.0,
                    "precision/class_03": 0.0,
                    "precision/class_04": 1.0,
                    "recall/class_00": 1.0,
                    "recall/class_01": 1.0,
                    "recall/class_02": 0.0,
                    "recall/class_03": 1.0,
                    "recall/class_04": 1.0,
                    "f1/class_00": 1.0,
                    "f1/class_01": 1.0,
                    "f1/class_02": 0.0,
                    "f1/class_03": 0.0,
                    "f1/class_04": 1.0,
                    "support/class_00": 1,
                    "support/class_01": 1,
                    "support/class_02": 1,
                    "support/class_03": 0,
                    "support/class_04": 1,
                },
            ],
            {
                "precision/_micro": 0.75,
                "recall/_micro": 0.75,
                "f1/_micro": 0.75,
                "precision/_macro": 0.8,
                "recall/_macro": 0.766667,
                "f1/_macro": 0.76,
                "precision/_weighted": 0.833333,
                "recall/_weighted": 0.75,
                "f1/_weighted": 0.766667,
                "precision/class_00": 1.0,
                "precision/class_01": 1.0,
                "precision/class_02": 0.333333,
                "precision/class_03": 0.666667,
                "precision/class_04": 1.0,
                "recall/class_00": 1.0,
                "recall/class_01": 0.666667,
                "recall/class_02": 0.5,
                "recall/class_03": 1.0,
                "recall/class_04": 0.666667,
                "f1/class_00": 1.0,
                "f1/class_01": 0.8,
                "f1/class_02": 0.4,
                "f1/class_03": 0.8,
                "f1/class_04": 0.8,
                "support/class_00": 2,
                "support/class_01": 3,
                "support/class_02": 2,
                "support/class_03": 2,
                "support/class_04": 3,
            },
        ),
    ),
)
def test_update_key_value_multiclass(
    outputs_list: Iterable[torch.Tensor],
    targets_list: Iterable[torch.Tensor],
    num_classes: int,
    zero_division: int,
    update_true_values: Iterable[Dict[str, float]],
    compute_true_value: Dict[str, float],
) -> None:
    """
    This test checks that metrics update works correctly with multiple calls.
    Metric should update statistics and return metrics for tmp input, so in this test
    we call update_key_value multiple times and check that all the intermediate metrics values
    are correct. After all the updates it checks that metrics computed with accumulated
    statistics are correct too.

    Args:
        outputs_list: sequence of predictions
        targets_list: sequence of targets
        num_classes: number of classes
        zero_division: int value, should be 0 or 1; return it in metrics in case of zero division
        update_true_values: sequence of true intermediate metrics
        compute_true_value: total metrics value for all the items from output_list and targets_list
    """
    metric = MulticlassPrecisionRecallF1SupportMetric(
        num_classes=num_classes, zero_division=zero_division
    )
    for outputs, targets, update_true_value in zip(outputs_list, targets_list, update_true_values):
        intermediate_metrics = metric.update_key_value(outputs=outputs, targets=targets)
        for key in update_true_value:
            assert key in intermediate_metrics
            assert abs(intermediate_metrics[key] - update_true_value[key]) < EPS
    metrics = metric.compute_key_value()
    for key in compute_true_value:
        assert key in metrics
        assert abs(metrics[key] - compute_true_value[key]) < EPS


@pytest.mark.parametrize(
    "outputs_list,targets_list,num_classes,zero_division,update_true_values,compute_true_value",
    (
        (
            [
                torch.tensor([[0, 1, 0, 1], [0, 0, 0, 0], [0, 1, 1, 0]]),
                torch.tensor([[0, 1, 1, 1], [0, 0, 0, 1], [0, 1, 0, 1]]),
                torch.tensor([[0, 1, 0, 0], [0, 1, 0, 1]]),
            ],
            [
                torch.tensor([[0, 1, 1, 1], [0, 0, 0, 0], [0, 1, 0, 1]]),
                torch.tensor([[0, 1, 0, 0], [0, 0, 1, 1], [1, 0, 1, 0]]),
                torch.tensor([[0, 1, 0, 0], [0, 0, 1, 0]]),
            ],
            4,
            1,
            [
                {
                    "precision/_micro": 0.75,
                    "recall/_micro": 0.6,
                    "f1/_micro": 0.666667,
                    "precision/_macro": 0.75,
                    "recall/_macro": 0.625,
                    "f1/_macro": 0.666667,
                    "precision/_weighted": 0.8,
                    "recall/_weighted": 0.6,
                    "f1/_weighted": 0.666667,
                    "precision/class_00": 1.0,
                    "precision/class_01": 1.0,
                    "precision/class_02": 0.0,
                    "precision/class_03": 1.0,
                    "recall/class_00": 1.0,
                    "recall/class_01": 1.0,
                    "recall/class_02": 0.0,
                    "recall/class_03": 0.5,
                    "f1/class_00": 1.0,
                    "f1/class_01": 1.0,
                    "f1/class_02": 0.0,
                    "f1/class_03": 0.666667,
                    "support/class_00": 0,
                    "support/class_01": 2,
                    "support/class_02": 1,
                    "support/class_03": 2,
                },
                {
                    "precision/_micro": 0.333333,
                    "recall/_micro": 0.4,
                    "f1/_micro": 0.363636,
                    "precision/_macro": 0.458333,
                    "recall/_macro": 0.5,
                    "f1/_macro": 0.291667,
                    "precision/_weighted": 0.366667,
                    "recall/_weighted": 0.4,
                    "f1/_weighted": 0.233333,
                    "precision/class_00": 1.0,
                    "precision/class_01": 0.5,
                    "precision/class_02": 0.0,
                    "precision/class_03": 0.333333,
                    "recall/class_00": 0.0,
                    "recall/class_01": 1.0,
                    "recall/class_02": 0.0,
                    "recall/class_03": 1.0,
                    "f1/class_00": 0.0,
                    "f1/class_01": 0.666667,
                    "f1/class_02": 0.0,
                    "f1/class_03": 0.5,
                    "support/class_00": 1,
                    "support/class_01": 1,
                    "support/class_02": 2,
                    "support/class_03": 1,
                },
                {
                    "precision/_micro": 0.333333,
                    "recall/_micro": 0.5,
                    "f1/_micro": 0.4,
                    "precision/_macro": 0.625,
                    "recall/_macro": 0.75,
                    "f1/_macro": 0.416667,
                    "precision/_weighted": 0.75,
                    "recall/_weighted": 0.5,
                    "f1/_weighted": 0.333333,
                    "precision/class_00": 1.0,
                    "precision/class_01": 0.5,
                    "precision/class_02": 1.0,
                    "precision/class_03": 0.0,
                    "recall/class_00": 1.0,
                    "recall/class_01": 1.0,
                    "recall/class_02": 0.0,
                    "recall/class_03": 1.0,
                    "f1/class_00": 1.0,
                    "f1/class_01": 0.666667,
                    "f1/class_02": 0.0,
                    "f1/class_03": 0.0,
                    "support/class_00": 0,
                    "support/class_01": 1,
                    "support/class_02": 1,
                    "support/class_03": 0,
                },
            ],
            {
                "precision/_micro": 0.461538,
                "recall/_micro": 0.5,
                "f1/_micro": 0.48,
                "precision/_macro": 0.516667,
                "recall/_macro": 0.416667,
                "f1/_macro": 0.325,
                "precision/_weighted": 0.405556,
                "recall/_weighted": 0.5,
                "f1/_weighted": 0.391667,
                "precision/class_00": 1.0,
                "precision/class_01": 0.666667,
                "precision/class_02": 0.0,
                "precision/class_03": 0.4,
                "recall/class_00": 0.0,
                "recall/class_01": 1.0,
                "recall/class_02": 0.0,
                "recall/class_03": 0.666667,
                "f1/class_00": 0.0,
                "f1/class_01": 0.8,
                "f1/class_02": 0.0,
                "f1/class_03": 0.5,
                "support/class_00": 1,
                "support/class_01": 4,
                "support/class_02": 4,
                "support/class_03": 3,
            },
        ),
    ),
)
def test_update_key_value_multilabel(
    outputs_list: Iterable[torch.Tensor],
    targets_list: Iterable[torch.Tensor],
    num_classes: int,
    zero_division: int,
    update_true_values: Iterable[Dict[str, float]],
    compute_true_value: Dict[str, float],
):
    """
    This test checks that metrics update works correctly with multiple calls.
    Metric should update statistics and return metrics for tmp input, so in this test
    we call update_key_value multiple times and check that all the intermediate metrics values
    are correct. After all the updates it checks that metrics computed with accumulated
    statistics are correct too.

    Args:
        outputs_list: sequence of predictions
        targets_list: sequence of targets
        num_classes: number of classes
        zero_division: int value, should be 0 or 1; return it in metrics in case of zero division
        update_true_values: sequence of true intermediate metrics
        compute_true_value: total metrics value for all the items from output_list and targets_list
    """
    metric = MultilabelPrecisionRecallF1SupportMetric(
        num_classes=num_classes, zero_division=zero_division
    )
    for outputs, targets, update_true_value in zip(outputs_list, targets_list, update_true_values):
        intermediate_metrics = metric.update_key_value(outputs=outputs, targets=targets)
        for key in update_true_value:
            assert key in intermediate_metrics
            assert abs(intermediate_metrics[key] - update_true_value[key]) < EPS
    metrics = metric.compute_key_value()
    for key in compute_true_value:
        assert key in metrics
        assert abs(metrics[key] - compute_true_value[key]) < EPS


@pytest.mark.parametrize(
    "outputs_list,targets_list,zero_division,update_true_values,compute_true_value",
    (
        (
            [torch.tensor([0, 1, 0, 1]), torch.tensor([0, 1, 0])],
            [torch.tensor([0, 0, 1, 1]), torch.tensor([0, 1, 1])],
            0,
            [
                {"precision": 0.5, "recall": 0.5, "f1": 0.5},
                {"precision": 1, "recall": 0.5, "f1": 0.666667},
            ],
            {"precision": 0.666667, "recall": 0.5, "f1": 0.571429},
        ),
    ),
)
def test_update_key_value_binary(
    outputs_list: Iterable[torch.Tensor],
    targets_list: Iterable[torch.Tensor],
    zero_division: int,
    update_true_values: Iterable[Dict[str, float]],
    compute_true_value: Dict[str, float],
):
    """
    This test checks that metrics update works correctly with multiple calls.
    Metric should update statistics and return metrics for tmp input, so in this test
    we call update_key_value multiple times and check that all the intermediate metrics values
    are correct. After all the updates it checks that metrics computed with accumulated
    statistics are correct too.

    Args:
        outputs_list: sequence of predictions
        targets_list: sequence of targets
        zero_division: int value, should be 0 or 1; return it in metrics in case of zero division
        update_true_values: sequence of true intermediate metrics
        compute_true_value: total metrics value for all the items from output_list and targets_list
    """
    metric = BinaryPrecisionRecallF1Metric(zero_division=zero_division)
    for outputs, targets, update_true_value in zip(outputs_list, targets_list, update_true_values):
        intermediate_metrics = metric.update_key_value(outputs=outputs, targets=targets)
        for key in update_true_value:
            assert key in intermediate_metrics
            assert abs(intermediate_metrics[key] - update_true_value[key]) < EPS
    metrics = metric.compute_key_value()
    for key in compute_true_value:
        assert key in metrics
        assert abs(metrics[key] - compute_true_value[key]) < EPS
