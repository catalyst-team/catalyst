# flake8: noqa
from typing import Dict, Iterable, Union

import pytest

import torch

from catalyst.metrics._r2_squared import R2Squared


@pytest.mark.parametrize(
    "outputs,targets,true_values",
    (
        (
            torch.Tensor([2.5, 0.0, 2, 8]),
            torch.Tensor([3, -0.5, 2, 7]),
            {
                "r2squared": torch.Tensor([0.9486]),
            },
        ),
    ),
)
def test_r2_squared(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    true_values: Dict[str, torch.Tensor],
) -> None:
    """
    Test r2 squared metric

    Args:
        outputs: tensor of outputs
        targets: tensor of targets
        true_values: true metric values
    """
    metric = R2Squared()
    metric.update(y_pred=outputs, y_true=targets)
    metrics = metric.compute_key_value()
    for key in true_values.keys():
        assert torch.isclose(true_values[key], metrics[key])


@pytest.mark.parametrize(
    "outputs_list,targets_list,true_values",
    (
        (
            (
                torch.Tensor([2.5, 0.0, 2, 8]),
                torch.Tensor([2.5, 0.0, 2, 8]),
                torch.Tensor([2.5, 0.0, 2, 8]),
                torch.Tensor([2.5, 0.0, 2, 8]),
            ),
            (
                torch.Tensor([3, -0.5, 2, 7]),
                torch.Tensor([3, -0.5, 2, 7]),
                torch.Tensor([3, -0.5, 2, 7]),
                torch.Tensor([3, -0.5, 2, 7]),
            ),
            {
                "r2squared": torch.Tensor([0.9486]),
            },
        ),
    ),
)
def test_r2_squared_update(
    outputs_list: Iterable[torch.Tensor],
    targets_list: Iterable[torch.Tensor],
    true_values: Dict[str, torch.Tensor],
):
    """
    Test r2 squared metric computation

    Args:
        outputs_list: list of outputs
        targets_list: list of targets
        true_values: true metric values
    """
    metric = R2Squared()
    for outputs, targets in zip(outputs_list, targets_list):
        metric.update(y_pred=outputs, y_true=targets)
    metrics = metric.compute_key_value()
    for key in true_values.keys():
        assert torch.isclose(true_values[key], metrics[key])
