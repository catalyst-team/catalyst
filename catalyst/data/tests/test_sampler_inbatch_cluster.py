from typing import List, Tuple

from pytest import mark

import torch

from catalyst.data.sampler_inbatch import HardClusterSampler

sampler = HardClusterSampler()


@mark.parametrize(
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
            torch.tensor(
                [
                    [True, False, False],
                    [False, True, False],
                    [False, False, True],
                ]
            ),
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
def test_get_labels_mask(labels: List[int], expected: torch.Tensor) -> None:
    """
    Test _get_labels_mask method of HardClusterSampler.

    Args:
        labels: list of labels -- input data for method _skip_diagonal
        expected: correct answer for labels input
    """
    labels_mask = sampler._get_labels_mask(labels)  # noqa: WPS437
    assert (labels_mask == expected).all()


@mark.parametrize(
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
                [
                    [[1, 1, 1], [1, 3, 1]],
                    [[2, 2, 6], [2, 6, 2]],
                    [[3, 3, 3], [3, 3, 9]],
                ],
                dtype=torch.float,
            ),
            torch.tensor([[[1, 1], [8, 8], [9, 9]]]),
        ],
    ],
)
def test_count_intra_class_distances(
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
    mean_vectors = features.mean(1)
    distances = sampler._count_intra_class_distances(  # noqa: WPS437
        features, mean_vectors
    )
    assert (distances == expected).all()


@mark.parametrize(
    ["mean_vectors", "expected"],
    [
        [
            torch.tensor(
                [[1, 0, 0, 0], [1, 1, 0, 0], [0, 1, 1, 0]], dtype=torch.float
            ),
            torch.tensor([[0, 1, 3], [1, 0, 2], [3, 2, 0]]),
        ],
        [
            torch.tensor(
                [[0, 0, 0, 0], [3, 0, 0, 0], [0, 4, 0, 0], [0, 0, 0, 5]],
                dtype=torch.float,
            ),
            torch.tensor(
                [
                    [0, 9, 16, 25],
                    [9, 0, 25, 34],
                    [16, 25, 0, 41],
                    [25, 34, 41, 0],
                ]
            ),
        ],
    ],
)
def test_count_inter_class_distances(mean_vectors, expected) -> None:
    """
    Test _count_inter_class_distances method of HardClusterSampler.

    Args:
        mean_vectors: tensor of shape (p, embed_dim) -- mean vectors of
        classes in the batch
        expected: tensor of shape (p, p) -- expected distances from mean
        vectors of classes
    """
    distances = sampler._count_inter_class_distances(  # noqa: WPS437
        mean_vectors
    )
    print(distances)
    assert (distances == expected).all()


@mark.parametrize(
    ["embed_dim", "labels", "expected_shape"],
    [
        [
            128,
            [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3],
            [(4, 128), (4, 128), (4, 128)],
        ],
        [32, [1, 2, 3, 4, 5, 1, 2, 3, 4, 5], [(5, 32), (5, 32), (5, 32)]],
    ],
)
def test_sample(
    embed_dim: int, labels: List[int], expected_shape: List[Tuple[int]]
) -> None:
    """
    Test output shapes in sample method of HardClusterSampler.

    Args:
        embed_dim: size of embedding
        labels: list of labels for samples in batch
        expected_shape: expected shape of output triplet
    """
    batch_size = len(labels)
    features = torch.rand(size=(batch_size, embed_dim))
    anchor, positive, negative = sampler.sample(features, labels)
    anchor_shape, pos_shape, neg_shape = expected_shape

    assert anchor.shape == anchor_shape
    assert positive.shape == pos_shape
    assert negative.shape == neg_shape
