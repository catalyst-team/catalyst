# flake8: noqa
from typing import List, Tuple
from collections import Counter
from random import randint

import numpy as np
import pytest
import torch
from torch import Tensor, tensor

from catalyst.data.sampler_inbatch import (
    AllTripletsSampler,
    HardClusterSampler,
    HardTripletsSampler,
    TLabels,
)
from catalyst.data.tests.test_sampler import generate_valid_labels
from catalyst.settings import SETTINGS
from catalyst.utils.misc import find_value_ids

if SETTINGS.ml_required:
    from scipy.spatial.distance import squareform
    from scipy.special import binom


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


def check_all_triplets_number(labels: List[int], num_selected_tri: int, max_tri: int) -> None:
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
    ids_anchor: List[int], ids_pos: List[int], ids_neg: List[int], labels: List[int],
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

        assert torch.isclose(distmat[i_a, ids_pos_cur].max(), distmat[i_a, i_p])

        assert torch.isclose(distmat[i_a, ids_neg_cur].min(), distmat[i_a, i_n])


@pytest.mark.skipif(not (SETTINGS.ml_required), reason="No scipy required")
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

        check_triplets_consistency(ids_anchor=ids_a, ids_pos=ids_p, ids_neg=ids_n, labels=labels)


@pytest.mark.skipif(not (SETTINGS.ml_required), reason="No scipy required")
def test_hard_sampler_from_features(features_and_labels) -> None:  # noqa: WPS442
    """
    Args:
        features_and_labels: features and valid labels
    """
    sampler = HardTripletsSampler(norm_required=True)

    for features, labels in features_and_labels:
        ids_a, ids_p, ids_n = sampler._sample(features=features, labels=labels)  # noqa: WPS437

        check_triplets_consistency(ids_anchor=ids_a, ids_pos=ids_p, ids_neg=ids_n, labels=labels)

        assert len(ids_a) == len(labels)


@pytest.mark.skipif(not (SETTINGS.ml_required), reason="No scipy required")
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
            ids_anchor=ids_a, ids_pos=ids_p, ids_neg=ids_n, labels=labels, distmat=distmat,
        )

        check_triplets_consistency(ids_anchor=ids_a, ids_pos=ids_p, ids_neg=ids_n, labels=labels)

        assert len(labels) == len(ids_a)


@pytest.mark.skipif(not (SETTINGS.ml_required), reason="No scipy required")
def test_hard_sampler_manual() -> None:
    """
    Test on manual example.
    """
    labels = [0, 0, 1, 1]

    dist_mat = torch.tensor(
        [[0.0, 0.3, 0.2, 0.4], [0.3, 0.0, 0.4, 0.8], [0.2, 0.4, 0.0, 0.5], [0.4, 0.8, 0.5, 0.0]]
    )

    gt = {(0, 1, 2), (1, 0, 2), (2, 3, 0), (3, 2, 0)}

    sampler = HardTripletsSampler(norm_required=True)

    ids_a, ids_p, ids_n = sampler._sample_from_distmat(  # noqa: WPS437
        distmat=dist_mat, labels=labels
    )
    predict = set(zip(ids_a, ids_p, ids_n))

    check_triplets_consistency(ids_anchor=ids_a, ids_pos=ids_p, ids_neg=ids_n, labels=labels)

    assert len(labels) == len(ids_a)
    assert predict == gt


@pytest.mark.skipif(not (SETTINGS.ml_required), reason="No scipy required")
@pytest.mark.parametrize(
    ["labels", "expected"],
    [
        [
            [0, 0, 1, 2, 2, 1],
            torch.tensor(
                [
                    [True, True, False, False, False, False],
                    [False, False, True, False, False, True],
                    [False, False, False, True, True, False],
                ]
            ),
        ],
        [
            [1, 2, 3],
            torch.tensor([[True, False, False], [False, True, False], [False, False, True]]),
        ],
        [
            [1, 1, 1, 1, 2, 2, 2, 2],
            torch.tensor(
                [
                    [True, True, True, True, False, False, False, False],
                    [False, False, False, False, True, True, True, True],
                ]
            ),
        ],
    ],
)
def test_cluster_get_labels_mask(labels: List[int], expected: torch.Tensor) -> None:
    """
    Test _get_labels_mask method of HardClusterSampler.

    Args:
        labels: list of labels -- input data for method _skip_diagonal
        expected: correct answer for labels input
    """
    sampler = HardClusterSampler()
    labels_mask = sampler._get_labels_mask(labels)  # noqa: WPS437
    assert (labels_mask == expected).all()


@pytest.mark.skipif(not (SETTINGS.ml_required), reason="No scipy required")
@pytest.mark.parametrize(
    ["features", "expected"],
    [
        [
            torch.tensor(
                [
                    [[1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 3]],
                    [[0, 3, 0, 1], [0, 6, 0, 1], [0, 3, 0, 1]],
                ],
                dtype=torch.float,
            ),
            torch.tensor([[1, 1, 4], [1, 4, 1]]),
        ],
        [
            torch.tensor(
                [[[1, 1, 1], [1, 3, 1]], [[2, 2, 6], [2, 6, 2]], [[3, 3, 3], [3, 3, 9]]],
                dtype=torch.float,
            ),
            torch.tensor([[[1, 1], [8, 8], [9, 9]]]),
        ],
    ],
)
def test_cluster_count_intra_class_distances(
    features: torch.Tensor, expected: torch.Tensor
) -> None:
    """
    Test _count_intra_class_distances method of HardClusterSampler.

    Args:
        features: tensor of shape (p, k, embed_dim), where p is a number of
        classes in the batch, k is a number of samples for each class,
        embed_dim is an embedding size -- features grouped by labels
        expected: tensor of shape (p, k) -- expected distances from mean
        vectors of classes to corresponding features
    """
    sampler = HardClusterSampler()
    mean_vectors = features.mean(1)
    distances = sampler._count_intra_class_distances(features, mean_vectors)  # noqa: WPS437
    assert (distances == expected).all()


@pytest.mark.skipif(not (SETTINGS.ml_required), reason="No scipy required")
@pytest.mark.parametrize(
    ["mean_vectors", "expected"],
    [
        [
            torch.tensor([[1, 0, 0, 0], [1, 1, 0, 0], [0, 1, 1, 0]], dtype=torch.float),
            torch.tensor([[0, 1, 3], [1, 0, 2], [3, 2, 0]], dtype=torch.float) ** 0.5,
        ],
        [
            torch.tensor(
                [[0, 0, 0, 0], [3, 0, 0, 0], [0, 4, 0, 0], [0, 0, 0, 5]], dtype=torch.float,
            ),
            torch.tensor(
                [[0, 9, 16, 25], [9, 0, 25, 34], [16, 25, 0, 41], [25, 34, 41, 0]],
                dtype=torch.float,
            )
            ** 0.5,
        ],
    ],
)
def test_cluster_count_inter_class_distances(mean_vectors, expected) -> None:
    """
    Test _count_inter_class_distances method of HardClusterSampler.

    Args:
        mean_vectors: tensor of shape (p, embed_dim) -- mean vectors of
        classes in the batch
        expected: tensor of shape (p, p) -- expected distances from mean
        vectors of classes
    """
    sampler = HardClusterSampler()
    distances = sampler._count_inter_class_distances(mean_vectors)  # noqa: WPS437
    assert (distances == expected).all()


@pytest.mark.skipif(not (SETTINGS.ml_required), reason="No scipy required")
@pytest.mark.parametrize(
    ["embed_dim", "labels", "expected_shape"],
    [
        [128, [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3], [(4, 128), (4, 128), (4, 128)]],
        [32, [1, 2, 3, 4, 5, 1, 2, 3, 4, 5], [(5, 32), (5, 32), (5, 32)]],
        [16, torch.tensor([0, 0, 1, 1]), [(2, 16), (2, 16), (2, 16)]],
    ],
)
def test_cluster_sample_shapes(
    embed_dim: int, labels: TLabels, expected_shape: List[Tuple[int]]
) -> None:
    """
    Test output shapes in sample method of HardClusterSampler.

    Args:
        embed_dim: size of embedding
        labels: list of labels for samples in batch
        expected_shape: expected shape of output triplet
    """
    sampler = HardClusterSampler()
    batch_size = len(labels)
    features = torch.rand(size=(batch_size, embed_dim))
    anchor, positive, negative = sampler.sample(features, labels)
    anchor_shape, pos_shape, neg_shape = expected_shape

    assert anchor.shape == anchor_shape
    assert positive.shape == pos_shape
    assert negative.shape == neg_shape


@pytest.mark.skipif(not (SETTINGS.ml_required), reason="No scipy required")
def test_triplet_cluster_edge_case() -> None:
    """
    Check an edge case of trivial samples for classes:
    expected HardTripletsSampler and HardClusterSampler to
    generate the same triplets.
    """
    features_dim = 128
    p, k = randint(2, 32), randint(2, 32)

    # Create a list of random labels
    unique_labels = torch.tensor(list(range(p)))
    # Create a list of random features for all the classes
    unique_features = torch.rand(size=(p, features_dim), dtype=torch.float)

    labels = unique_labels.repeat((k))
    features = unique_features.repeat((k, 1))

    hard_triplet_sampler = HardTripletsSampler()
    hard_cluster_sampler = HardClusterSampler()

    triplets = hard_triplet_sampler.sample(features, labels)
    cluster_triplets = hard_cluster_sampler.sample(features, labels)

    # Concatenates tensors from triplets to use torch.unique for comparison
    triplets = torch.cat(triplets, dim=1)
    cluster_triplets = torch.cat(cluster_triplets, dim=1)

    triplets = torch.unique(triplets, dim=0)
    cluster_triplets = torch.unique(cluster_triplets, dim=0)

    assert torch.allclose(triplets, cluster_triplets, atol=1e-10)
