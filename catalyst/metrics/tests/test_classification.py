from typing import Dict, Iterable

import pytest
import torch

from catalyst.metrics import (
    BinaryPrecisionRecallF1SupportMetric,
    MulticlassPrecisionRecallF1SupportMetric,
    MultilabelPrecisionRecallF1SupportMetric,
)

EPS = 1e-4


@pytest.mark.parametrize(
    "outputs,targets,num_classes,zero_division,true_values",
    (
        (
            torch.tensor([1, 0, 3, 2, 2, 2]),
            torch.tensor([0, 0, 2, 4, 3, 1]),
            5,
            1,
            {
                "precision/macro": 0.4,
                "precision/micro": 0.166667,
                "precision/weighted": 0.5,
                "precision/class_01": 1.0,
                "precision/class_02": 0.0,
                "precision/class_03": 0.0,
                "precision/class_04": 0.0,
                "precision/class_05": 1.0,
                "recall/macro": 0.1,
                "recall/micro": 0.166667,
                "recall/weighted": 0.166667,
                "recall/class_01": 0.5,
                "recall/class_02": 0.0,
                "recall/class_03": 0.0,
                "recall/class_04": 0.0,
                "recall/class_05": 0.0,
                "f1/macro": 0.133333,
                "f1/micro": 0.166667,
                "f1/weighted": 0.222222,
                "f1/class_01": 0.666667,
                "f1/class_02": 0.0,
                "f1/class_03": 0.0,
                "f1/class_04": 0.0,
                "f1/class_05": 0.0,
                "support/class_01": 2,
                "support/class_02": 1,
                "support/class_03": 1,
                "support/class_04": 1,
                "support/class_05": 1,
            },
        ),
        (
            torch.tensor([2, 2, 2, 3, 3, 4]),
            torch.tensor([4, 4, 2, 1, 3, 1]),
            5,
            0,
            {
                "precision/macro": 0.166667,
                "precision/micro": 0.333333,
                "precision/weighted": 0.138889,
                "precision/class_01": 0.0,
                "precision/class_02": 0.0,
                "precision/class_03": 0.333333,
                "precision/class_04": 0.5,
                "precision/class_05": 0.0,
                "recall/macro": 0.4,
                "recall/micro": 0.333333,
                "recall/weighted": 0.333333,
                "recall/class_01": 0.0,
                "recall/class_02": 0.0,
                "recall/class_03": 1.0,
                "recall/class_04": 1.0,
                "recall/class_05": 0.0,
                "f1/macro": 0.233333,
                "f1/micro": 0.333333,
                "f1/weighted": 0.194444,
                "f1/class_01": 0.0,
                "f1/class_02": 0.0,
                "f1/class_03": 0.5,
                "f1/class_04": 0.666667,
                "f1/class_05": 0.0,
                "support/class_01": 0,
                "support/class_02": 2,
                "support/class_03": 1,
                "support/class_04": 1,
                "support/class_05": 2,
            },
        ),
        (
            torch.tensor([2, 2, 2, 3, 3, 4]),
            torch.tensor([4, 4, 2, 1, 3, 1]),
            5,
            1,
            {
                "precision/macro": 0.566667,
                "precision/micro": 0.333333,
                "precision/weighted": 0.472222,
                "precision/class_01": 1.0,
                "precision/class_02": 1.0,
                "precision/class_03": 0.333333,
                "precision/class_04": 0.5,
                "precision/class_05": 0.0,
                "recall/macro": 0.6,
                "recall/micro": 0.333333,
                "recall/weighted": 0.333333,
                "recall/class_01": 1.0,
                "recall/class_02": 0.0,
                "recall/class_03": 1.0,
                "recall/class_04": 1.0,
                "recall/class_05": 0.0,
                "f1/macro": 0.433333,
                "f1/micro": 0.333333,
                "f1/weighted": 0.194444,
                "f1/class_01": 1.0,
                "f1/class_02": 0.0,
                "f1/class_03": 0.5,
                "f1/class_04": 0.666667,
                "f1/class_05": 0.0,
                "support/class_01": 0,
                "support/class_02": 2,
                "support/class_03": 1,
                "support/class_04": 1,
                "support/class_05": 2,
            },
        ),
        (
            torch.tensor([5, 1, 4, 0, 4, 6, 2, 2, 0, 5]),
            torch.tensor([1, 2, 1, 1, 2, 5, 2, 0, 6, 6]),
            7,
            1,
            {
                "precision/macro": 0.214286,
                "precision/micro": 0.1,
                "precision/weighted": 0.15,
                "precision/class_01": 0.0,
                "precision/class_02": 0.0,
                "precision/class_03": 0.5,
                "precision/class_04": 1.0,
                "precision/class_05": 0.0,
                "precision/class_06": 0.0,
                "precision/class_07": 0.0,
                "recall/macro": 0.333333,
                "recall/micro": 0.1,
                "recall/weighted": 0.1,
                "recall/class_01": 0.0,
                "recall/class_02": 0.0,
                "recall/class_03": 0.333333,
                "recall/class_04": 1.0,
                "recall/class_05": 1.0,
                "recall/class_06": 0.0,
                "recall/class_07": 0.0,
                "f1/macro": 0.2,
                "f1/micro": 0.1,
                "f1/weighted": 0.12,
                "f1/class_01": 0.0,
                "f1/class_02": 0.0,
                "f1/class_03": 0.4,
                "f1/class_04": 1.0,
                "f1/class_05": 0.0,
                "f1/class_06": 0.0,
                "f1/class_07": 0.0,
                "support/class_01": 1,
                "support/class_02": 3,
                "support/class_03": 3,
                "support/class_04": 0,
                "support/class_05": 0,
                "support/class_06": 1,
                "support/class_07": 2,
            },
        ),
        (
            torch.tensor([2, 2, 1, 1, 2, 2, 1, 2, 2, 0]),
            torch.tensor([1, 2, 0, 2, 2, 1, 1, 2, 0, 2]),
            3,
            0,
            {
                "precision/macro": 0.277778,
                "precision/micro": 0.4,
                "precision/weighted": 0.35,
                "precision/class_01": 0.0,
                "precision/class_02": 0.333333,
                "precision/class_03": 0.5,
                "recall/macro": 0.311111,
                "recall/micro": 0.4,
                "recall/weighted": 0.4,
                "recall/class_01": 0.0,
                "recall/class_02": 0.333333,
                "recall/class_03": 0.6,
                "f1/macro": 0.292929,
                "f1/micro": 0.4,
                "f1/weighted": 0.372727,
                "f1/class_01": 0.0,
                "f1/class_02": 0.333333,
                "f1/class_03": 0.545455,
                "support/class_01": 2,
                "support/class_02": 3,
                "support/class_03": 5,
            },
        ),
        (
            torch.tensor([2, 0, 0, 0]),
            torch.tensor([2, 2, 0, 2]),
            3,
            0,
            {
                "precision/macro": 0.444444,
                "precision/micro": 0.5,
                "precision/weighted": 0.833333,
                "precision/class_01": 0.333333,
                "precision/class_02": 0.0,
                "precision/class_03": 1.0,
                "recall/macro": 0.444444,
                "recall/micro": 0.5,
                "recall/weighted": 0.5,
                "recall/class_01": 1.0,
                "recall/class_02": 0.0,
                "recall/class_03": 0.333333,
                "f1/macro": 0.333333,
                "f1/micro": 0.5,
                "f1/weighted": 0.5,
                "f1/class_01": 0.5,
                "f1/class_02": 0.0,
                "f1/class_03": 0.5,
                "support/class_01": 1,
                "support/class_02": 0,
                "support/class_03": 3,
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
    metrics = metric(outputs=outputs, targets=targets)
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
                "precision/macro": 0.833333,
                "precision/micro": 0.833333,
                "precision/weighted": 0.75,
                "precision/class_01": 1.0,
                "precision/class_02": 1.0,
                "precision/class_03": 0.5,
                "recall/macro": 0.75,
                "recall/micro": 0.625,
                "recall/weighted": 0.625,
                "recall/class_01": 1.0,
                "recall/class_02": 1.0,
                "recall/class_03": 0.25,
                "f1/macro": 0.777778,
                "f1/micro": 0.714286,
                "f1/weighted": 0.666667,
                "f1/class_01": 1.0,
                "f1/class_02": 1.0,
                "f1/class_03": 0.333333,
                "support/class_01": 1,
                "support/class_02": 3,
                "support/class_03": 4,
            },
        ),
        (
            torch.tensor([[0, 1, 0, 0], [1, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0], [0, 1, 1, 0],]),
            torch.tensor([[0, 1, 1, 1], [1, 1, 1, 0], [0, 0, 0, 0], [0, 0, 1, 1], [0, 1, 1, 0],]),
            4,
            0,
            {
                "precision/macro": 0.625,
                "precision/micro": 0.833333,
                "precision/weighted": 0.6,
                "precision/class_01": 1.0,
                "precision/class_02": 1.0,
                "precision/class_03": 0.5,
                "precision/class_04": 0.0,
                "recall/macro": 0.5625,
                "recall/micro": 0.5,
                "recall/weighted": 0.5,
                "recall/class_01": 1.0,
                "recall/class_02": 1.0,
                "recall/class_03": 0.25,
                "recall/class_04": 0.0,
                "f1/macro": 0.583333,
                "f1/micro": 0.625,
                "f1/weighted": 0.533333,
                "f1/class_01": 1.0,
                "f1/class_02": 1.0,
                "f1/class_03": 0.333333,
                "f1/class_04": 0.0,
                "support/class_01": 1,
                "support/class_02": 3,
                "support/class_03": 4,
                "support/class_04": 2,
            },
        ),
        (
            torch.tensor([[0, 1, 0, 0], [1, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0], [0, 1, 1, 0],]),
            torch.tensor([[0, 1, 1, 1], [1, 1, 1, 0], [0, 0, 0, 0], [0, 0, 1, 1], [0, 1, 1, 0],]),
            4,
            1,
            {
                "precision/macro": 0.875,
                "precision/micro": 0.833333,
                "precision/weighted": 0.8,
                "precision/class_01": 1.0,
                "precision/class_02": 1.0,
                "precision/class_03": 0.5,
                "precision/class_04": 1.0,
                "recall/macro": 0.5625,
                "recall/micro": 0.5,
                "recall/weighted": 0.5,
                "recall/class_01": 1.0,
                "recall/class_02": 1.0,
                "recall/class_03": 0.25,
                "recall/class_04": 0.0,
                "f1/macro": 0.583333,
                "f1/micro": 0.625,
                "f1/weighted": 0.533333,
                "f1/class_01": 1.0,
                "f1/class_02": 1.0,
                "f1/class_03": 0.333333,
                "f1/class_04": 0.0,
                "support/class_01": 1,
                "support/class_02": 3,
                "support/class_03": 4,
                "support/class_04": 2,
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
                "precision/macro": 0.533333,
                "precision/micro": 0.611111,
                "precision/weighted": 0.516667,
                "precision/class_01": 1.0,
                "precision/class_02": 0.833333,
                "precision/class_03": 0.5,
                "precision/class_04": 0.0,
                "precision/class_05": 0.333333,
                "recall/macro": 0.657143,
                "recall/micro": 0.55,
                "recall/weighted": 0.55,
                "recall/class_01": 1.0,
                "recall/class_02": 1.0,
                "recall/class_03": 0.285714,
                "recall/class_04": 0.0,
                "recall/class_05": 1.0,
                "f1/macro": 0.554545,
                "f1/micro": 0.578947,
                "f1/weighted": 0.504545,
                "f1/class_01": 1.0,
                "f1/class_02": 0.909091,
                "f1/class_03": 0.363636,
                "f1/class_04": 0.0,
                "f1/class_05": 0.5,
                "support/class_01": 2,
                "support/class_02": 5,
                "support/class_03": 7,
                "support/class_04": 4,
                "support/class_05": 2,
            },
        ),
        (
            torch.tensor([[0, 1], [1, 0], [0, 1], [0, 0], [1, 1],]),
            torch.tensor([[1, 1], [1, 1], [0, 0], [0, 1], [1, 1]]),
            2,
            0,
            {
                "precision/macro": 0.833333,
                "precision/micro": 0.8,
                "precision/weighted": 0.809524,
                "precision/class_01": 1.0,
                "precision/class_02": 0.666667,
                "recall/macro": 0.583333,
                "recall/micro": 0.571429,
                "recall/weighted": 0.571429,
                "recall/class_01": 0.666667,
                "recall/class_02": 0.5,
                "f1/macro": 0.685714,
                "f1/micro": 0.666667,
                "f1/weighted": 0.669388,
                "f1/class_01": 0.8,
                "f1/class_02": 0.571429,
                "support/class_01": 3,
                "support/class_02": 4,
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
    metrics = metric(outputs=outputs, targets=targets)
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
            {"precision": 0.75, "recall": 0.6, "f1": 0.666667,},
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
            {"precision": 0, "recall": 0, "f1": 0,},
        ),
        (
            torch.tensor([1, 1, 1, 0]),
            torch.tensor([0, 0, 0, 0]),
            1,
            {"precision": 0, "recall": 1, "f1": 0,},
        ),
    ),
)
def test_binary_metrics(
    outputs: torch.Tensor, targets: torch.Tensor, zero_division: int, true_values: Dict[str, float]
) -> None:
    """
    Test binary metric
    Args:
        outputs: tensor of predictions
        targets: tensor of targets
        zero_division: zero division policy flag
        true_values: true values of metrics
    """
    metric = BinaryPrecisionRecallF1SupportMetric(zero_division=zero_division)
    metrics = metric(outputs=outputs, targets=targets)
    for key in true_values:
        assert key in metrics
        assert abs(metrics[key] - true_values[key]) < EPS


@pytest.mark.parametrize(
    "outputs_list,targets_list,num_classes,zero_division,true_values",
    (
        (
            [torch.tensor([1, 2, 3]), torch.tensor([0, 3, 4]), torch.tensor([4, 5]),],
            [torch.tensor([1, 2, 4]), torch.tensor([0, 3, 4]), torch.tensor([5, 5]),],
            6,
            0,
            {
                "precision/macro": 0.833333,
                "precision/micro": 0.75,
                "precision/weighted": 0.8125,
                "precision/class_01": 1.0,
                "precision/class_02": 1.0,
                "precision/class_03": 1.0,
                "precision/class_04": 0.5,
                "precision/class_05": 0.5,
                "precision/class_06": 1.0,
                "recall/macro": 0.833333,
                "recall/micro": 0.75,
                "recall/weighted": 0.75,
                "recall/class_01": 1.0,
                "recall/class_02": 1.0,
                "recall/class_03": 1.0,
                "recall/class_04": 1.0,
                "recall/class_05": 0.5,
                "recall/class_06": 0.5,
                "f1/macro": 0.805556,
                "f1/micro": 0.75,
                "f1/weighted": 0.75,
                "f1/class_01": 1.0,
                "f1/class_02": 1.0,
                "f1/class_03": 1.0,
                "f1/class_04": 0.666667,
                "f1/class_05": 0.5,
                "f1/class_06": 0.666667,
                "support/class_01": 1,
                "support/class_02": 1,
                "support/class_03": 1,
                "support/class_04": 1,
                "support/class_05": 2,
                "support/class_06": 2,
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
