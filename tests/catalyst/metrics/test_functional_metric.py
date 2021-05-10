# flake8: noqa
from typing import Any, Callable, Dict

import pytest
import torch

from catalyst.metrics import FunctionalBatchMetric

base_outputs = torch.tensor([[0.8, 0.1, 0], [0, 0.4, 0.3], [0, 0, 1]])
base_targets = torch.tensor([[1.0, 0, 0], [0, 1, 0], [1, 1, 0]])
base_outputs_1 = torch.stack([base_outputs, base_targets])[None, :, :, :]
base_targets_1 = torch.stack([base_targets, base_targets])[None, :, :, :]
base_outputs_2 = torch.stack([base_outputs, base_targets])[None, :, :, :]
base_targets_2 = torch.stack([base_targets, base_outputs])[None, :, :, :]
EPS = 1e-5


def custom_dice(outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    intersection = torch.sum(outputs * targets)
    union = outputs.sum() + targets.sum()
    return 2 * intersection / union


@pytest.mark.parametrize(
    "outputs_1, targets_1, outputs_2, targets_2, metric_function, prefix, batch_answer_1, "
    "batch_answer_2, total_answer",
    (
        (
            torch.Tensor([12, 3.1, 12, 0, -1]),
            torch.Tensor([12.1, 2.9, 10, -1, 4]),
            torch.Tensor([2, 3.1, 13, -1]),
            torch.Tensor([1.1, 2.9, -1, 3.1]),
            torch.nn.functional.l1_loss,
            "custom_mae",
            {"custom_mae": torch.tensor(1.6600)},
            {"custom_mae": torch.tensor(4.8000)},
            {"custom_mae/mean": torch.tensor(3.0555556)},
        ),
        (
            torch.Tensor([12, 3.1, 12, 0, -1]),
            torch.Tensor([12.1, 2.9, 10, -1, 4]),
            torch.Tensor([2, 3.1, 13, -1]),
            torch.Tensor([1.1, 2.9, -1, 3.1]),
            torch.nn.functional.mse_loss,
            "custom_mse",
            {"custom_mse": torch.tensor(6.0100)},
            {"custom_mse": torch.tensor(53.4150)},
            {"custom_mse/mean": torch.tensor(27.078888)},
        ),
        (
            base_outputs_1,
            base_targets_1,
            base_outputs_2,
            base_targets_2,
            custom_dice,
            "custom_dice",
            {"custom_dice": torch.tensor(0.71232873)},
            {"custom_dice": torch.tensor(0.36363637)},
            {"custom_dice/mean": torch.tensor(0.5379826)},
        ),
    ),
)
def test_mae_metric(
    outputs_1: torch.Tensor,
    targets_1: torch.Tensor,
    outputs_2: torch.Tensor,
    targets_2: torch.Tensor,
    metric_function: Callable,
    prefix: str,
    batch_answer_1: Dict[str, Any],
    batch_answer_2: Dict[str, Any],
    total_answer: Dict[str, Any],
):
    metric = FunctionalBatchMetric(metric_fn=metric_function, metric_key=prefix)
    batch_score_1 = metric.update_key_value(len(outputs_1), outputs_1, targets_1)
    batch_score_2 = metric.update_key_value(len(outputs_2), outputs_2, targets_2)
    loader_metric = metric.compute_key_value()
    for key, value in batch_answer_1.items():
        assert key in batch_score_1
        assert abs(batch_score_1[key] - batch_answer_1[key]) < EPS
    for key, value in batch_answer_2.items():
        assert key in batch_score_2
        assert abs(batch_score_2[key] - batch_answer_2[key]) < EPS
    for key, value in total_answer.items():
        assert key in loader_metric
        assert abs(loader_metric[key] - total_answer[key]) < EPS
