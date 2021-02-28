# flake8: noqa
from typing import Iterable

import numpy as np
import pytest

from catalyst.metrics._additive import AdditiveValueMetric


@pytest.mark.parametrize(
    "values_list,num_samples_list,true_values_list",
    (
        ([1, 2, 3, 4, 5], [100, 200, 300, 400, 500], [1, 1.666667, 2.333333, 3, 3.666667]),
        ([1, 0, 2, 3], [10, 5, 15, 25], [1, 0.666667, 1.333333, 2.090909]),
        (
            [100, 10, 1000, 10000, 0],
            [14, 43, 555, 32, 9],
            [100, 32.105263, 909.852941, 1361.537267, 1342.771822],
        ),
    ),
)
def test_additive_mean(
    values_list: Iterable[float],
    num_samples_list: Iterable[int],
    true_values_list: Iterable[float],
) -> None:
    """
    Test additive metric mean computation

    Args:
        values_list: list of values to update metric
        num_samples_list: list of num_samples
        true_values_list: list of metric intermediate value
    """
    metric = AdditiveValueMetric()
    for value, num_samples, true_value in zip(values_list, num_samples_list, true_values_list):
        metric.update(value=value, num_samples=num_samples)
        mean, _ = metric.compute()
        assert np.isclose(mean, true_value)


@pytest.mark.parametrize(
    "values_list,num_samples_list,true_values_list",
    (
        ([1, 2, 3, 4, 5], [100, 200, 300, 400, 500], [0, 0.472192, 0.745978, 1.0005, 1.247635]),
        ([1, 0, 2, 3], [10, 5, 15, 25], [0, 0.48795, 0.758098, 1.005038]),
        (
            [100, 10, 1000, 10000, 0],
            [14, 43, 555, 32, 9],
            [0, 39.084928, 281.772757, 1995.83843, 1988.371749],
        ),
    ),
)
def test_additive_std(
    values_list: Iterable[float],
    num_samples_list: Iterable[int],
    true_values_list: Iterable[float],
):
    """
    Test additive metric std computation

    Args:
        values_list: list of values to update metric
        num_samples_list: list of num_samples
        true_values_list: list of metric intermediate value
    """
    metric = AdditiveValueMetric()
    for value, num_samples, true_value in zip(values_list, num_samples_list, true_values_list):
        metric.update(value=value, num_samples=num_samples)
        _, std = metric.compute()
        assert np.isclose(std, true_value)
