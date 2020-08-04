# flake8: noqa
# @TODO: code formatting issue for 20.07 release
from typing import List, Tuple
from collections import Counter

import numpy as np
import pytest
from scipy.spatial.distance import squareform
from scipy.special import binom

import torch
from torch import Tensor, tensor

from catalyst.contrib.utils.misc import find_value_ids
from catalyst.data.sampler_inbatch import (
    AllTripletsSampler,
    HardTripletsSampler,
)
from catalyst.data.tests.test_sampler import generate_valid_labels


@pytest.fixture()
def features_and_labels() -> List[Tuple[Tensor, List[int]]]:
    """
    Returns: list of features and valid labels
    """
    num_batches = 100
    features_dim = 10

    labels_pk = generate_valid_labels(num=num_batches)
    labels_list, _, _ = zip(*labels_pk)

    features = []
    for labels in labels_list:
        features.append(torch.rand(size=(len(labels), features_dim)))

    return list(zip(features, labels_list))


@pytest.fixture()
def distmats_and_labels() -> List[Tuple[Tensor, List[int]]]:
    """
    Returns: list of distance matrices and valid labels
    """
    num_batches = 100

    labels_pk = generate_valid_labels(num=num_batches)
    labels_list, _, _ = zip(*labels_pk)

    distmats = []
    for labels in labels_list:
        n = len(labels)
        distmats.append(tensor(squareform(torch.rand(int(n * (n - 1) / 2)))))

    return list(zip(distmats, labels_list))


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
    """
    labels_counts = Counter(labels).values()

    n_all_tri = 0
    for count in labels_counts:
        n_pos = binom(count, 2)
        n_neg = len(labels) - count
        n_all_tri += n_pos * n_neg

    assert num_selected_tri == n_all_tri or num_selected_tri == max_tri


def check_triplets_consistency(
    ids_anchor: List[int],
    ids_pos: List[int],
    ids_neg: List[int],
    labels: List[int],
) -> None:
    """
    Args:
        ids_anchor: anchor indexes of selected triplets
        ids_pos: positive indexes of selected triplets
        ids_neg: negative indexes of selected triplets
        labels: labels of the samples in the batch
    """
    num_sampled_tri = len(ids_anchor)

    assert num_sampled_tri == len(ids_pos) == len(ids_neg)

    for (i_a, i_p, i_n) in zip(ids_anchor, ids_pos, ids_neg):
        assert len({i_a, i_p, i_n}) == 3
        assert labels[i_a] == labels[i_p]
        assert labels[i_a] != labels[i_n]

    unq_tri = set(zip(ids_anchor, ids_pos, ids_neg))

    assert num_sampled_tri == len(unq_tri)


def check_triplets_are_hardest(
    ids_anchor: List[int],
    ids_pos: List[int],
    ids_neg: List[int],
    labels: List[int],
    distmat: Tensor,
) -> None:
    """
    Args:
        ids_anchor: anchor indexes of selected triplets
        ids_pos: positive indexes of selected triplets
        ids_neg: negative indexes of selected triplets
        labels: labels of the samples in the batch
        distmat: distances between features
    """
    ids_all = set(range(len(labels)))

    for i_a, i_p, i_n in zip(ids_anchor, ids_pos, ids_neg):
        ids_label = set(find_value_ids(it=labels, value=labels[i_a]))

        ids_pos_cur = np.array(list(ids_label - {i_a}), int)
        ids_neg_cur = np.array(list(ids_all - ids_label), int)

        assert torch.isclose(
            distmat[i_a, ids_pos_cur].max(), distmat[i_a, i_p]
        )

        assert torch.isclose(
            distmat[i_a, ids_neg_cur].min(), distmat[i_a, i_n]
        )


def test_all_triplets_sampler(features_and_labels) -> None:  # noqa: WPS442
    """
    Args:
        features_and_labels: features and valid labels
    """
    max_tri = 512
    sampler = AllTripletsSampler(max_output_triplets=max_tri)

    for _, labels in features_and_labels:  # noqa: WPS437
        ids_a, ids_p, ids_n = sampler._sample(labels=labels)  # noqa: WPS437

        check_all_triplets_number(
            labels=labels, max_tri=max_tri, num_selected_tri=len(ids_a),
        )

        check_triplets_consistency(
            ids_anchor=ids_a, ids_pos=ids_p, ids_neg=ids_n, labels=labels
        )


def test_hard_sampler_from_features(
    features_and_labels,  # noqa: WPS442
) -> None:
    """
    Args:
        features_and_labels: features and valid labels
    """
    sampler = HardTripletsSampler(norm_required=True)

    for features, labels in features_and_labels:
        ids_a, ids_p, ids_n = sampler._sample(  # noqa: WPS437
            features=features, labels=labels
        )

        check_triplets_consistency(
            ids_anchor=ids_a, ids_pos=ids_p, ids_neg=ids_n, labels=labels
        )

        assert len(ids_a) == len(labels)


def test_hard_sampler_from_dist(distmats_and_labels) -> None:  # noqa: WPS442
    """
    Args:
        distmats_and_labels:
            list of distance matrices and valid labels
    """
    sampler = HardTripletsSampler(norm_required=True)

    for distmat, labels in distmats_and_labels:
        ids_a, ids_p, ids_n = sampler._sample_from_distmat(  # noqa: WPS437
            distmat=distmat, labels=labels
        )

        check_triplets_are_hardest(
            ids_anchor=ids_a,
            ids_pos=ids_p,
            ids_neg=ids_n,
            labels=labels,
            distmat=distmat,
        )

        check_triplets_consistency(
            ids_anchor=ids_a, ids_pos=ids_p, ids_neg=ids_n, labels=labels
        )

        assert len(labels) == len(ids_a)


def test_hard_sampler_manual() -> None:
    """
    Test on manual example.
    """
    labels = [0, 0, 1, 1]

    dist_mat = torch.tensor(
        [
            [0.0, 0.3, 0.2, 0.4],
            [0.3, 0.0, 0.4, 0.8],
            [0.2, 0.4, 0.0, 0.5],
            [0.4, 0.8, 0.5, 0.0],
        ]
    )

    gt = {(0, 1, 2), (1, 0, 2), (2, 3, 0), (3, 2, 0)}

    sampler = HardTripletsSampler(norm_required=True)

    ids_a, ids_p, ids_n = sampler._sample_from_distmat(  # noqa: WPS437
        distmat=dist_mat, labels=labels
    )
    predict = set(zip(ids_a, ids_p, ids_n))

    check_triplets_consistency(
        ids_anchor=ids_a, ids_pos=ids_p, ids_neg=ids_n, labels=labels
    )

    assert len(labels) == len(ids_a)
    assert predict == gt
