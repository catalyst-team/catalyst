from typing import List, Tuple
from collections import Counter
from operator import itemgetter
from random import randint, shuffle

import pytest

from catalyst.data.sampler import BalanceBatchSampler


@pytest.fixture()
def input_balance_batch_sampler() -> List[Tuple[List[int], int, int]]:
    """
    Returns: test data for sampler in the following order: (labels, p, k)
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

    num_random_cases = 0
    # (alekseysh) It was checked once with N = 100_000 before doing the PR
    for _ in range(num_random_cases):
        # code below generates same valid inputs for sampler
        p, k = randint(2, 12), randint(2, 12)
        labels_ = [[label] * randint(2, 12) for label in range(p + 1)]
        labels = [el for sublist in labels_ for el in sublist]
        shuffle(labels)
        input_cases.append((labels, p, k))

    return input_cases


def single_check_balance_batch_sampler(
    labels: List[int], p: int, k: int
) -> None:
    """
    Args:
        labels: list of classes labels
        p: number of classes in a batch
        k: number of instances for each class in a batch

    Returns: None
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
        assert 4 <= len(set(batch_ids))

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


def test_balance_batch_sampler(input_balance_batch_sampler) -> None:
    """
    Args:
        input_balance_batch_sampler: pytest fixture

    Returns: None
    """
    for labels, p, k in input_balance_batch_sampler:
        single_check_balance_batch_sampler(labels, p, k)
