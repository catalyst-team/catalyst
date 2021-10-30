# flake8: noqa
from typing import Dict, List, Union

import pytest

import torch

from catalyst.metrics import DiceMetric, IOUMetric, TrevskyMetric

base_outputs = torch.tensor([[0.8, 0.1, 0], [0, 0.4, 0.3], [0, 0, 1]])
base_targets = torch.tensor([[1.0, 0, 0], [0, 1, 0], [1, 1, 0]])
base_outputs = torch.stack([base_outputs, base_targets])[None, :, :, :]
base_targets = torch.stack([base_targets, base_targets])[None, :, :, :]

base_outputs_2 = torch.tensor([[0.8, 0.1, 0.4], [0.1, 0.4, 0.3], [0, 1, 1]])
base_targets_2 = torch.tensor([[1.0, 0.1, 0], [0, 0.5, 0], [0, 1, 1]])
base_outputs_2 = torch.stack([base_outputs_2, base_targets_2])[None, :, :, :]
base_targets_2 = torch.stack([base_targets_2, base_targets_2])[None, :, :, :]

EPS = 1e-5


@pytest.mark.parametrize(
    "outputs, targets, weights, class_names, batch_answers, total_answers",
    (
        (
            [base_outputs, base_outputs_2],
            [base_targets, base_targets_2],
            [0.2, 0.8],
            ["class_name_00", "class_name_01"],
            [
                {
                    "dice/class_name_00": 0.3636363446712494,
                    "dice/class_name_01": 1.0,
                    "dice": 0.6818182,
                    "dice/_weighted": 0.8727272748947144,
                },
                {
                    "dice/class_name_00": 0.781818151473999,
                    "dice/class_name_01": 0.9055555462837219,
                    "dice": 0.8436868190765381,
                    "dice/_weighted": 0.8808081150054932,
                },
            ],
            [
                {
                    "dice/class_name_00": 0.3636363446712494,
                    "dice/class_name_01": 1.0,
                    "dice": 0.6818181872367859,
                    "dice/_micro": 0.7123287916183472,
                    "dice/_weighted": 0.8727272748947144,
                },
                {
                    "dice/class_name_00": 0.5888112187385559,
                    "dice/class_name_01": 0.9552631378173828,
                    "dice/_micro": 0.7776271104812622,
                    "dice": 0.772037148475647,
                    "dice/_macro": 0.772037148475647,
                    "dice/_weighted": 0.8819727897644043,
                },
            ],
        ),
    ),
)
def test_dice_metric(
    outputs: List[torch.Tensor],
    targets: List[torch.Tensor],
    weights: List[float],
    class_names: List[str],
    batch_answers: List[Dict[str, float]],
    total_answers: List[Dict[str, float]],
):
    """Docs."""
    metric = DiceMetric(weights=weights, class_names=class_names)
    for output, target, batch_answer, total_answer in zip(
        outputs, targets, batch_answers, total_answers
    ):
        batch_score = metric.update_key_value(output, target)
        total_score = metric.compute_key_value()
        for key, value in batch_answer.items():
            assert key in batch_score
            assert abs(batch_score[key] - batch_answer[key]) < EPS
        for key, value in total_answer.items():
            assert key in total_score
            assert abs(total_score[key] - total_answer[key]) < EPS


@pytest.mark.parametrize(
    "outputs, targets, weights, class_names, batch_answers, total_answers",
    (
        (
            [base_outputs, base_outputs_2],
            [base_targets, base_targets_2],
            [0.2, 0.8],
            ["class_name_00", "class_name_01"],
            [[0.3636363446712494, 1.0], [0.781818151473999, 0.9055555462837219]],
            [
                [
                    [0.3636363446712494, 1.0],
                    0.7123287916183472,
                    0.6818181872367859,
                    0.8727272748947144,
                ],
                [
                    [0.5888112187385559, 0.9552631378173828],
                    0.7776271104812622,
                    0.772037148475647,
                    0.8819727897644043,
                ],
            ],
        ),
    ),
)
def test_dice_metric_compute(
    outputs: List[torch.Tensor],
    targets: List[torch.Tensor],
    weights: List[float],
    class_names: List[str],
    batch_answers: List[List[float]],
    total_answers: List[List[Union[List[float], float]]],
):
    """Docs."""
    metric = DiceMetric(weights=weights, class_names=class_names)
    for output, target, batch_answer, total_answer in zip(
        outputs, targets, batch_answers, total_answers
    ):
        batch_score = metric.update(output, target)
        total_score = metric.compute()
        assert len(batch_answer) == len(batch_score)
        for pred, answer in zip(batch_score, batch_answer):
            assert abs(pred - answer) < EPS
        assert len(total_score) == len(total_score)
        for pred, answer in zip(total_score, total_score):
            if isinstance(pred, list):
                for pred_sample, answer_sample in zip(pred, answer):
                    assert abs(pred_sample - answer_sample) < EPS
            else:
                assert abs(pred - answer) < EPS


@pytest.mark.parametrize(
    "outputs, targets, weights, class_names, batch_answers, total_answers",
    (
        (
            [base_outputs, base_outputs_2],
            [base_targets, base_targets_2],
            [0.2, 0.8],
            ["class_name_00", "class_name_01"],
            [
                {
                    "iou/class_name_00": 0.2222222536802292,
                    "iou/class_name_01": 1.0,
                    "iou": 0.6111111,
                    "iou/_weighted": 0.8444444537162781,
                },
                {
                    "iou/class_name_00": 0.641791045665741,
                    "iou/class_name_01": 0.8274111747741699,
                    "iou": 0.7346011400222778,
                    "iou/_weighted": 0.7902871370315552,
                },
            ],
            [
                {
                    "iou/class_name_00": 0.2222222536802292,
                    "iou/class_name_01": 1.0,
                    "iou": 0.6111111044883728,
                    "iou/_micro": 0.5531914830207825,
                    "iou/_weighted": 0.8444444537162781,
                },
                {
                    "iou/class_name_00": 0.4172447919845581,
                    "iou/class_name_01": 0.9143576622009277,
                    "iou/_micro": 0.6361619234085083,
                    "iou": 0.6658012270927429,
                    "iou/_macro": 0.6658012270927429,
                    "iou/_weighted": 0.8149350881576538,
                },
            ],
        ),
    ),
)
def test_iou_metric(
    outputs: List[torch.Tensor],
    targets: List[torch.Tensor],
    weights: List[float],
    class_names: List[str],
    batch_answers: List[Dict[str, float]],
    total_answers: List[Dict[str, float]],
):
    """Docs."""
    metric = IOUMetric(weights=weights, class_names=class_names)
    for output, target, batch_answer, total_answer in zip(
        outputs, targets, batch_answers, total_answers
    ):
        batch_score = metric.update_key_value(output, target)
        total_score = metric.compute_key_value()
        for key, value in batch_answer.items():
            assert key in batch_score
            assert abs(batch_score[key] - batch_answer[key]) < EPS
        for key, value in total_answer.items():
            assert key in total_score
            assert abs(total_score[key] - total_answer[key]) < EPS


@pytest.mark.parametrize(
    "outputs, targets, weights, class_names, batch_answers, total_answers",
    (
        (
            [base_outputs, base_outputs_2],
            [base_targets, base_targets_2],
            [0.2, 0.8],
            ["class_name_00", "class_name_01"],
            [[0.2222222536802292, 1.0], [0.641791045665741, 0.8274111747741699]],
            [
                [
                    [0.2222222536802292, 1.0],
                    0.5531914830207825,
                    0.6111111044883728,
                    0.8444444537162781,
                ],
                [
                    [0.4172447919845581, 0.9143576622009277],
                    0.6361619234085083,
                    0.6658012270927429,
                    0.8149350881576538,
                ],
            ],
        ),
    ),
)
def test_iou_metric_compute(
    outputs: List[torch.Tensor],
    targets: List[torch.Tensor],
    weights: List[float],
    class_names: List[str],
    batch_answers: List[List[float]],
    total_answers: List[List[Union[List[float], float]]],
):
    """IOU update, compute test"""
    metric = IOUMetric(weights=weights, class_names=class_names)
    for output, target, batch_answer, total_answer in zip(
        outputs, targets, batch_answers, total_answers
    ):
        batch_score = metric.update(output, target)
        total_score = metric.compute()
        assert len(batch_answer) == len(batch_score)
        for pred, answer in zip(batch_score, batch_answer):
            assert abs(pred - answer) < EPS
        assert len(total_score) == len(total_score)
        for pred, answer in zip(total_score, total_score):
            if isinstance(pred, list):
                for pred_sample, answer_sample in zip(pred, answer):
                    assert abs(pred_sample - answer_sample) < EPS
            else:
                assert abs(pred - answer) < EPS


@pytest.mark.parametrize(
    "outputs, targets, alpha, weights, class_names, batch_answers, total_answers",
    (
        (
            [base_outputs, base_outputs_2],
            [base_targets, base_targets_2],
            0.2,
            [0.2, 0.8],
            ["class_name_00", "class_name_01"],
            [
                {
                    "trevsky/class_name_00": 0.4166666567325592,
                    "trevsky/class_name_01": 1.0,
                    "trevsky": 0.7083333134651184,
                    "trevsky/_weighted": 0.8833333253860474,
                },
                {
                    "trevsky/class_name_00": 0.7524999976158142,
                    "trevsky/class_name_01": 0.9055555462837219,
                    "trevsky": 0.8290277719497681,
                    "trevsky/_weighted": 0.8749444484710693,
                },
            ],
            [
                {
                    "trevsky/class_name_00": 0.4166666567325592,
                    "trevsky/class_name_01": 1.0,
                    "trevsky": 0.7083333134651184,
                    "trevsky/_micro": 0.7558139562606812,
                    "trevsky/_weighted": 0.8833333253860474,
                },
                {
                    "trevsky/class_name_00": 0.6119186282157898,
                    "trevsky/class_name_01": 0.9552631974220276,
                    "trevsky/_micro": 0.7921270728111267,
                    "trevsky": 0.7835909128189087,
                    "trevsky/_macro": 0.7835909128189087,
                    "trevsky/_weighted": 0.886594295501709,
                },
            ],
        ),
    ),
)
def test_trevsky_metric(
    outputs: List[torch.Tensor],
    targets: List[torch.Tensor],
    alpha: float,
    weights: List[float],
    class_names: List[str],
    batch_answers: List[Dict[str, float]],
    total_answers: List[Dict[str, float]],
):
    metric = TrevskyMetric(alpha=alpha, weights=weights, class_names=class_names)
    for output, target, batch_answer, total_answer in zip(
        outputs, targets, batch_answers, total_answers
    ):
        batch_score = metric.update_key_value(output, target)
        total_score = metric.compute_key_value()
        for key, value in batch_answer.items():
            assert key in batch_score
            assert abs(batch_score[key] - batch_answer[key]) < EPS
        for key, value in total_answer.items():
            assert key in total_score
            assert abs(total_score[key] - total_answer[key]) < EPS


@pytest.mark.parametrize(
    "outputs, targets, alpha, weights, class_names, batch_answers, total_answers",
    (
        (
            [base_outputs, base_outputs_2],
            [base_targets, base_targets_2],
            0.2,
            [0.2, 0.8],
            ["class_name_00", "class_name_01"],
            [[0.4166666567325592, 1.0], [0.7524999976158142, 0.9055555462837219]],
            [
                [
                    [0.4166666567325592, 1.0],
                    0.7558139562606812,
                    0.7083333134651184,
                    0.8833333253860474,
                ],
                [
                    [0.6119186282157898, 0.9552631974220276],
                    0.7921270728111267,
                    0.7835909128189087,
                    0.886594295501709,
                ],
            ],
        ),
    ),
)
def test_trevsky_metric_compute(
    outputs: List[torch.Tensor],
    targets: List[torch.Tensor],
    alpha: float,
    weights: List[float],
    class_names: List[str],
    batch_answers: List[List[float]],
    total_answers: List[List[Union[List[float], float]]],
):
    """Trevsky update, compute test"""
    metric = TrevskyMetric(alpha=alpha, weights=weights, class_names=class_names)
    for output, target, batch_answer, total_answer in zip(
        outputs, targets, batch_answers, total_answers
    ):
        batch_score = metric.update(output, target)
        total_score = metric.compute()
        assert len(batch_answer) == len(batch_score)
        for pred, answer in zip(batch_score, batch_answer):
            assert abs(pred - answer) < EPS
        assert len(total_score) == len(total_score)
        for pred, answer in zip(total_score, total_score):
            if isinstance(pred, list):
                for pred_sample, answer_sample in zip(pred, answer):
                    assert abs(pred_sample - answer_sample) < EPS
            else:
                assert abs(pred - answer) < EPS
