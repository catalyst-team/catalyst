# flake8: noqa
from typing import Dict, List

import pytest
import torch

from catalyst.metrics import DiceMetric, IOUMetric, TrevskyMetric

base_outputs = torch.tensor([[0.8, 0.1, 0], [0, 0.4, 0.3], [0, 0, 1]])
base_targets = torch.tensor([[1.0, 0, 0], [0, 1, 0], [1, 1, 0]])
base_outputs = torch.stack([base_outputs, base_targets])[None, :, :, :]
base_targets = torch.stack([base_targets, base_targets])[None, :, :, :]
EPS = 1e-5


@pytest.mark.parametrize(
    "outputs, targets, weights, class_names, batch_answer, total_answer",
    (
        (
            base_outputs,
            base_targets,
            [0.2, 0.8],
            ["class_name_00", "class_name_01"],
            {
                "dice/class_name_00": 0.3636363446712494,
                "dice/class_name_01": 1.0,
                "dice": 0.6818182,
                "dice/_weighted": 0.8727272748947144,
            },
            {
                "dice/class_name_00": 0.3636363446712494,
                "dice/class_name_01": 1.0,
                "dice": 0.6818181872367859,
                "dice/_micro": 0.7123287916183472,
                "dice/_weighted": 0.8727272748947144,
            },
        ),
    ),
)
def test_dice_metric(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    weights: List[float],
    class_names: List[str],
    batch_answer: Dict[str, float],
    total_answer: Dict[str, float],
):
    """Docs."""
    metric = DiceMetric(weights=weights, class_names=class_names)
    batch_score = metric.update_key_value(outputs, targets)
    total_score = metric.compute_key_value()
    for key, value in batch_answer.items():
        assert key in batch_score
        assert abs(batch_score[key] - batch_answer[key]) < EPS
    for key, value in total_answer.items():
        assert key in total_score
        assert abs(total_score[key] - total_answer[key]) < EPS


@pytest.mark.parametrize(
    "outputs, targets, weights, class_names, batch_answer, total_answer",
    (
        (
            base_outputs,
            base_targets,
            [0.2, 0.8],
            ["class_name_00", "class_name_01"],
            {
                "iou/class_name_00": 0.2222222536802292,
                "iou/class_name_01": 1.0,
                "iou": 0.6111111,
                "iou/_weighted": 0.8444444537162781,
            },
            {
                "iou/class_name_00": 0.2222222536802292,
                "iou/class_name_01": 1.0,
                "iou": 0.6111111044883728,
                "iou/_micro": 0.5531914830207825,
                "iou/_weighted": 0.8444444537162781,
            },
        ),
    ),
)
def test_iou_metric(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    weights: List[float],
    class_names: List[str],
    batch_answer: Dict[str, float],
    total_answer: Dict[str, float],
):
    """Docs."""
    metric = IOUMetric(weights=weights, class_names=class_names)
    batch_score = metric.update_key_value(outputs, targets)
    total_score = metric.compute_key_value()
    for key, value in batch_answer.items():
        assert key in batch_score
        assert abs(batch_score[key] - batch_answer[key]) < EPS
    for key, value in total_answer.items():
        assert key in total_score
        assert abs(total_score[key] - total_answer[key]) < EPS


@pytest.mark.parametrize(
    "outputs, targets, alpha, weights, class_names, batch_answer, total_answer",
    (
        (
            base_outputs,
            base_targets,
            0.2,
            [0.2, 0.8],
            ["class_name_00", "class_name_01"],
            {
                "trevsky/class_name_00": 0.4166666567325592,
                "trevsky/class_name_01": 1.0,
                "trevsky": 0.7083333134651184,
                "trevsky/_weighted": 0.8833333253860474,
            },
            {
                "trevsky/class_name_00": 0.4166666567325592,
                "trevsky/class_name_01": 1.0,
                "trevsky": 0.7083333134651184,
                "trevsky/_micro": 0.7558139562606812,
                "trevsky/_weighted": 0.8833333253860474,
            },
        ),
    ),
)
def test_trevsky_metric(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float,
    weights: List[float],
    class_names: List[str],
    batch_answer: Dict[str, float],
    total_answer: Dict[str, float],
):
    metric = TrevskyMetric(alpha=alpha, weights=weights, class_names=class_names)
    batch_score = metric.update_key_value(outputs, targets)
    total_score = metric.compute_key_value()
    for key, value in batch_answer.items():
        assert key in batch_score
        assert abs(batch_score[key] - batch_answer[key]) < EPS
    for key, value in total_answer.items():
        assert key in total_score
        assert abs(total_score[key] - total_answer[key]) < EPS
