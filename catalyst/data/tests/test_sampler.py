from collections import Counter
from operator import itemgetter
from random import randint, shuffle
from typing import List, Tuple

import pytest
from scipy.special import binom

from catalyst.data.sampler import AllTripletsSampler, BalanceBatchSampler

TLabelsPK = List[Tuple[List[int], int, int]]


def generate_labels_pk(num: int) -> TLabelsPK:
    """
    This function generates same valid inputs for samplers.
    It generates k instances for p classes.

    Args:
        num: number of generated samples

    Returns: samples in the folowing order: (labels, p, k)
    """
    labels_pk = []

    for _ in range(num):
        p, k = randint(2, 12), randint(2, 12)
        labels_ = [[label] * randint(2, 12) for label in range(p)]
        labels = [el for sublist in labels_ for el in sublist]

        shuffle(labels)
        labels_pk.append((labels, p, k))

    return labels_pk


@pytest.fixture()
def input_for_inbatch_sampler() -> List[int]:
    """
    Returns: list of valid classes labels
    """
    labels_pk = generate_labels_pk(num=100)
    labels, _, _ = zip(*labels_pk)
    return labels


@pytest.fixture()
def input_for_balance_batch_sampler() -> TLabelsPK:
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

    # (alekseysh) It was checked once with N = 100_000 before doing the PR
    num_random_cases = 0
    input_cases.extend((generate_pk_labels(num_random_cases)))

    return input_cases


def check_balance_batch_sampler_epoch(
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


def test_balance_batch_sampler(input_for_balance_batch_sampler) -> None:
    """
    Args:
        input_balance_batch_sampler: pytest fixture

    Returns: None
    """
    for labels, p, k in input_for_balance_batch_sampler:
        check_balance_batch_sampler_epoch(labels, p, k)


def check_all_triplets_number(
    labels: List[int], num_selected_tri: int, max_tri: int
) -> None:
    """
    Checks that the selection strategy for all triplets
    returns the correct number of triplets.

    Args:
        labels: list of classes labels
        num_selected_tri: number of selected triplets
        max_tri: limit on the number of selected triplets

    Returns: None
    """
    counts = Counter(labels).values()

    n_all_tri = 0
    for count in counts:
        n_pos = binom(count, 2)
        n_neg = len(labels) - count
        n_all_tri += n_pos * n_neg

    assert n_all_tri == num_selected_tri or max_tri == num_selected_tri


def check_triplets_consistency(
    ids_anchor: List[int],
    ids_pos: List[int],
    ids_neg: List[int],
    labels: List[int],
) -> None:
    """
    Args:
        ids_anchor: anchor indeces of selected triplets
        ids_pos: positive indeces of selected triplets
        ids_neg: negative indeces of selected triplets
        labels: labels of the samples in the batch

    Returns: None
    """
    assert len(ids_anchor) == len(ids_pos) == len(ids_neg)

    for (i_a, i_p, i_n) in zip(ids_anchor, ids_pos, ids_neg):
        assert len({i_a, i_p, i_n}) == 3
        assert labels[i_a] == labels[i_p]
        assert labels[i_a] != labels[i_n]

    n_unq_tri = len(set(list(zip(ids_anchor, ids_pos, ids_neg))))
    assert len(ids_anchor) == n_unq_tri


def test_all_triplets_sampler(input_for_inbatch_sampler) -> None:
    """
    Args:
        input_for_inbatch_sampler: list of valid labels

    Returns: None
    """
    max_tri = 512
    sampler = AllTripletsSampler(max_output_triplets=max_tri)

    for labels in input_for_inbatch_sampler:
        ids_a, ids_p, ids_n = sampler._sample(labels=labels)

        check_all_triplets_number(
            labels=labels, max_tri=max_tri, num_selected_tri=len(ids_a),
        )

        check_triplets_consistency(
            ids_anchor=ids_a, ids_pos=ids_p, ids_neg=ids_n, labels=labels
        )


test_all_triplets_sampler(input_for_inbatch_sampler())
