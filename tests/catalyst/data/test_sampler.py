from typing import List, Tuple
from collections import Counter
from operator import itemgetter
from random import randint, shuffle

import numpy as np
import pytest

from catalyst.data.sampler import BalanceBatchSampler, DynamicBalanceClassSampler

TLabelsPK = List[Tuple[List[int], int, int]]


def generate_valid_labels(num: int) -> TLabelsPK:
    """
    This function generates some valid inputs for samplers.
    It generates k instances for p classes.

    Args:
        num: number of generated samples

    Returns:
        samples in the folowing order: (labels, p, k)
    """
    labels_pk = []

    for _ in range(num):  # noqa: WPS122
        p, k = randint(2, 12), randint(2, 12)
        labels_list = [[label] * randint(2, 12) for label in range(p)]
        labels = [el for sublist in labels_list for el in sublist]

        shuffle(labels)
        labels_pk.append((labels, p, k))

    return labels_pk


@pytest.fixture()
def input_for_balance_batch_sampler() -> TLabelsPK:
    """
    Returns:
        test data for sampler in the following order: (labels, p, k)
    """
    input_cases = [
        # ideal case
        ([0, 1, 2, 3, 0, 1, 2, 3], 2, 2),
        # repetation sampling is needed for class #3
        ([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2], 2, 3),
        # check last batch behaviour:
        # last batch includes less than p classes (2 < 3)
        ([0, 1, 2, 3, 4, 0, 1, 2, 3, 4], 3, 2),
        # we need to drop 1 class during the epoch because
        # number of classes in data % p = 1
        ([0, 1, 2, 3, 0, 1, 2, 3], 3, 2),
        # several random cases
        ([0, 1, 2, 2, 1, 0, 1, 0, 2, 0, 1, 2], 3, 5),
        ([0, 1, 2, 2, 1, 0, 1, 0, 2, 0, 1, 2], 2, 3),
        ([0, 1, 2, 2, 1, 0, 1, 0, 2, 0, 1, 2], 3, 2),
    ]

    # (alekseysh) It was checked once with N = 100_000 before doing the PR
    num_random_cases = 100
    input_cases.extend((generate_valid_labels(num_random_cases)))

    return input_cases


def check_balance_batch_sampler_epoch(labels: List[int], p: int, k: int) -> None:
    """
    Args:
        labels: list of classes labels
        p: number of classes in a batch
        k: number of instances for each class in a batch
    """
    sampler = BalanceBatchSampler(labels=labels, p=p, k=k)
    sampled_ids = list(sampler)

    sampled_classes = []
    # emulating of 1 epoch
    for i in range(sampler.batches_in_epoch):
        i_batch_start = i * sampler.batch_size
        i_batch_end = min((i + 1) * sampler.batch_size, len(sampler) + 1)
        batch_ids = sampled_ids[i_batch_start:i_batch_end]
        batch_labels = itemgetter(*batch_ids)(labels)

        labels_counter = Counter(batch_labels)
        num_batch_classes = len(labels_counter)
        num_batch_instances = list(labels_counter.values())
        cur_batch_size = len(batch_labels)
        sampled_classes.extend(list(labels_counter.keys()))

        # batch-level invariants
        assert len(set(batch_ids)) >= 4

        is_last_batch = i == sampler.batches_in_epoch - 1
        if is_last_batch:
            assert 1 < num_batch_classes <= p
            assert all(1 < el <= k for el in num_batch_instances)
            assert 2 * 2 <= cur_batch_size <= p * k
        else:
            assert num_batch_classes == p
            assert all(el == k for el in num_batch_instances)
            assert cur_batch_size == p * k

    # epoch-level invariants
    num_classes_in_data = len(set(labels))
    num_classes_in_epoch = len(set(sampled_classes))
    assert (num_classes_in_data == num_classes_in_epoch) or (
        num_classes_in_data == num_classes_in_epoch + 1
    )

    assert max(sampled_ids) <= len(labels) - 1


def test_balance_batch_sampler(input_for_balance_batch_sampler) -> None:  # noqa: WPS442
    """
    Args:
        input_for_balance_batch_sampler: list of (labels, p, k)
    """
    for labels, p, k in input_for_balance_batch_sampler:
        check_balance_batch_sampler_epoch(labels=labels, p=p, k=k)


@pytest.fixture()
def input_for_dynamic_balance_class_sampler() -> List[Tuple[list, float]]:
    """
    This function generates some valid inputs for DynamicBalanceClassSampler

    Returns:
        inputs in the folowing order: (labels, exp_l)
    """
    labels = [
        # class imbalance
        np.array([0] * 100 + [1] * 10 + [2] * 20),
        # uniform class distribution
        np.array([0] * 10 + [1] * 10 + [2] * 10),
        # random class distribution
        np.random.randint(0, 4, size=(200,)),
    ]
    exp_lambda = np.linspace(0.1, 0.95, 11)
    input_cases = np.transpose(
        [np.tile(labels, len(exp_lambda)), np.repeat(exp_lambda, len(labels))]
    )
    return input_cases


def check_dynamic_balance_class_sampler(labels: List, exp_l: float) -> None:
    """
    Check DynamicBalanceClassSampler on certain inputs

    Args:
        labels: list of labels
        exp_l: exponent figure for schedule
    """
    sampler = DynamicBalanceClassSampler(labels, exp_l)
    n_labels = len(np.unique(labels))
    labels_counter = Counter(labels)
    min_class_key, min_class_size = labels_counter.most_common(n_labels)[-1]
    current_d = {key: value / min_class_size for key, value in Counter(labels).items()}
    for _ in range(10):  # noqa: WPS122
        new_counter = Counter(labels[list(sampler.__iter__())])
        new_d = {key: value / min_class_size for key, value in new_counter.items()}
        for key, value in new_d.items():
            assert value <= current_d[key]
        assert new_d[min_class_key] == 1
        current_d = new_d


def test_dynamic_balance_class_sampler(
    input_for_dynamic_balance_class_sampler,  # noqa: WPS442
) -> None:
    """
    Tests for DynamicBalanceClassSampler

    Args:
        input_for_dynamic_balance_class_sampler: list of (labels, exp_l)
    """
    for labels, exp_l in input_for_dynamic_balance_class_sampler:
        check_dynamic_balance_class_sampler(labels, exp_l)
