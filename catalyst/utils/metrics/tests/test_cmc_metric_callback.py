import torch

from catalyst.contrib.dl.callbacks.cmc_callback import (  # noqa: F401
    CMCScoreCallback,
)
from catalyst.utils.metrics.cmc_score import _cmc_score_count


def test_metric_simple_0():
    """Simple test with two examples"""
    distances = torch.tensor([[1, 2], [2, 1]])
    conformity_matrix = torch.tensor([[0, 1], [1, 0]])
    out = _cmc_score_count(
        distances=distances, conformity_matrix=conformity_matrix
    )
    expected = 0.0
    assert out == expected


def test_metric_simple_05():
    """Simple test with two examples"""
    distances = torch.tensor([[0, 0.5], [0.0, 0.5]])
    conformity_matrix = torch.tensor([[0, 1], [1, 0]])
    out = _cmc_score_count(
        distances=distances, conformity_matrix=conformity_matrix
    )
    expected = 0.5
    assert out == expected


def test_metric_simple_1():
    """Simple test with two examples"""
    distances = torch.tensor([[1, 0.5], [0.5, 1]])
    conformity_matrix = torch.tensor([[0, 1], [1, 0]])
    out = _cmc_score_count(
        distances=distances, conformity_matrix=conformity_matrix
    )
    expected = 1.0
    assert out == expected


def test_metric_simple_1_k_2():
    """Simple test with topk=2"""
    distances = torch.tensor([[1, 2], [2, 1]])
    conformity_matrix = torch.tensor([[0, 1], [1, 0]])
    out = _cmc_score_count(
        distances=distances, conformity_matrix=conformity_matrix, topk=2
    )
    expected = 1.0
    assert out == expected
