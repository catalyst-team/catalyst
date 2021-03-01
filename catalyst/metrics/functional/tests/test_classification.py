# flake8: noqa
from typing import Tuple

import numpy as np
import pytest

from catalyst.metrics.functional._classification import (
    f1score,
    get_aggregated_metrics,
    precision,
    recall,
)

EPS = 1e-5


@pytest.mark.parametrize(
    "tp,fp,zero_division,true_value",
    ((5, 3, 1, 0.625), (5, 5, 0, 0.5), (0, 0, 0, 0), (0, 0, 1, 1)),
)
def test_precision(tp: int, fp: int, zero_division: int, true_value: float):
    """
    Test precision metric

    Args:
        tp: true positive statistic
        fp: false positive statistic
        zero_division: 0 or 1, value to return in case of zero division
        true_value: true metric value
    """
    precision_value = precision(tp=tp, fp=fp, zero_division=zero_division)
    assert (precision_value - true_value) < EPS


@pytest.mark.parametrize(
    "tp,fn,zero_division,true_value",
    ((5, 3, 1, 0.625), (5, 5, 0, 0.5), (0, 0, 0, 0), (0, 0, 1, 1)),
)
def test_recall(tp: int, fn: int, zero_division: int, true_value: float):
    """
    Test recall metric

    Args:
        tp: true positive statistic
        fn: false negative statistic
        zero_division: 0 or 1, value to return in case of zero division
        true_value: true metric value
    """
    recall_value = recall(tp=tp, fn=fn, zero_division=zero_division)
    assert (recall_value - true_value) < EPS


@pytest.mark.parametrize(
    "precision_value,recall_value,true_value",
    ((0.8, 0.7, 0.746667), (0.5, 0.5, 0.5), (0.6, 0.4, 0.48)),
)
def test_f1score(precision_value: float, recall_value: float, true_value: float):
    """
    Test f1 score

    Args:
        precision_value: precision value
        recall_value: recall value
        true_value: true metric value
    """
    f1 = f1score(precision_value=precision_value, recall_value=recall_value)
    assert abs(f1 - true_value) < EPS


@pytest.mark.parametrize(
    "tp,fp,fn,support,zero_division,true_answer",
    (
        (
            np.array([0.0, 1.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 0.0, 1.0]),
            np.array([1.0, 0.0, 0.0, 0.0]),
            np.array([1.0, 1.0, 1.0, 0.0]),
            1,
            (0.666667, 0.666667, 0.666667),
        ),
        (
            np.array([1.0, 2.0, 2.0, 0.0]),
            np.array([0.0, 0.0, 0.0, 2.0]),
            np.array([2.0, 0.0, 0.0, 0.0]),
            np.array([3.0, 2.0, 2.0, 0.0]),
            0,
            (0.714286, 0.714286, 0.714286),
        ),
        (
            np.array([1.0, 2.0, 2.0, 0.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 0.0, 2.0, 1.0, 1.0]),
            np.array([3.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
            np.array([4.0, 2.0, 2.0, 0.0, 0.0, 1.0]),
            0,
            (0.555556, 0.555556, 0.555556),
        ),
    ),
)
def test_micro(
    tp: np.array,
    fp: np.array,
    fn: np.array,
    support: np.array,
    zero_division: int,
    true_answer: Tuple[float],
):
    """
    Test micro metrics averaging

    Args:
        tp: true positive statistic
        fp: false positive statistic
        fn: false negative statistic
        support: support statistic
        zero_division: 0 or 1
        true_answer: true metric value
    """
    _, micro, _, _ = get_aggregated_metrics(
        tp=tp, fp=fp, fn=fn, support=support, zero_division=zero_division
    )
    assert micro[-1] is None
    for pred, real in zip(micro[:-1], true_answer):
        assert abs(pred - real) < EPS


@pytest.mark.parametrize(
    "tp,fp,fn,support,zero_division,true_answer",
    (
        (
            np.array([0, 1, 3]),
            np.array([1, 2, 3]),
            np.array([2, 2, 2]),
            np.array([2, 3, 5]),
            0,
            (0.277778, 0.311111, 0.292929),
        ),
        (
            np.array([1.0, 2.0, 1.0, 0.0, 1.0, 1.0]),
            np.array([0.0, 0.0, 1.0, 1.0, 0.0, 0.0]),
            np.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
            np.array([2.0, 3.0, 1.0, 0.0, 1.0, 1.0]),
            0,
            (0.75, 0.694444, 0.688889),
        ),
        (
            np.array([0.0, 1.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 0.0, 1.0]),
            np.array([1.0, 0.0, 0.0, 0.0]),
            np.array([1.0, 1.0, 1.0, 0.0]),
            1,
            (0.75, 0.75, 0.5),
        ),
    ),
)
def test_macro_average(
    tp: np.array,
    fp: np.array,
    fn: np.array,
    support: np.array,
    zero_division: int,
    true_answer: Tuple[float],
):
    """
    Test macro metrics averaging

    Args:
        tp: true positive statistic
        fp: false positive statistic
        fn: false negative statistic
        support: support statistic
        zero_division: 0 or 1
        true_answer: true metric value
    """
    _, _, macro, _ = get_aggregated_metrics(
        tp=tp, fp=fp, fn=fn, support=support, zero_division=zero_division
    )
    assert macro[-1] is None
    for pred, real in zip(macro[:-1], true_answer):
        assert abs(pred - real) < EPS


@pytest.mark.parametrize(
    "tp,fp,fn,support,zero_division,true_answer",
    (
        (
            np.array([1.0, 2.0, 2.0, 0.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 0.0, 2.0, 1.0, 1.0]),
            np.array([3.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
            np.array([4.0, 2.0, 2.0, 0.0, 0.0, 1.0]),
            0,
            (0.888889, 0.555556, 0.622222),
        ),
        (
            np.array([1.0, 2.0, 2.0, 0.0]),
            np.array([0.0, 0.0, 0.0, 2.0]),
            np.array([2.0, 0.0, 0.0, 0.0]),
            np.array([3.0, 2.0, 2.0, 0.0]),
            0,
            (1.0, 0.714286, 0.785714),
        ),
        (
            np.array([0.0, 1.0]),
            np.array([1.0, 1.0]),
            np.array([1.0, 1.0]),
            np.array([1.0, 2.0]),
            0,
            (0.333333, 0.333333, 0.333333),
        ),
    ),
)
def test_weighted(
    tp: np.array,
    fp: np.array,
    fn: np.array,
    support: np.array,
    zero_division: int,
    true_answer: Tuple[float],
):
    """
    Test weighted metrics averaging

    Args:
        tp: true positive statistic
        fp: false positive statistic
        fn: false negative statistic
        support: support statistic
        zero_division: 0 or 1
        true_answer: true metric value
    """
    _, _, _, weighted = get_aggregated_metrics(
        tp=tp, fp=fp, fn=fn, support=support, zero_division=zero_division
    )
    assert weighted[-1] is None
    for pred, real in zip(weighted[:-1], true_answer):
        assert abs(pred - real) < EPS
