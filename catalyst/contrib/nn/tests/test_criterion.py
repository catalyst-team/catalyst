import torch
import numpy as np
from catalyst.contrib.nn import criterion as module
from catalyst.contrib.nn.criterion import (
    CircleLoss,
    TripletMarginLossWithSampler,
    CosFaceLoss,
)
from catalyst.data import AllTripletsSampler


def test_criterion_init():
    """@TODO: Docs. Contribution is welcome."""
    for module_class in module.__dict__.values():
        if isinstance(module_class, type):
            if module_class == CircleLoss:
                instance = module_class(margin=0.25, gamma=256)
            elif module_class == TripletMarginLossWithSampler:
                instance = module_class(
                    margin=1.0, sampler_inbatch=AllTripletsSampler()
                )
            else:
                instance = module_class()
            assert instance is not None


def test_cosface_loss():
    emb_size = 4
    n_classes = 3
    s = 3.0
    m = 0.1
    loss_fn = CosFaceLoss(emb_size, n_classes, s, m)

    features = np.array(
        [
            [1, 2, 3, 4],
            [5, 6, 7, 8],
        ],
        dtype="f",
    )
    target = np.array([0, 2], dtype="l")
    projection = np.array(
        [
            [0.1, 0.2, 0.3, 0.4],
            [1.1, 3.2, 5.3, 0.4],
            [0.1, 0.2, 6.3, 0.4],
        ],
        dtype="f",
    )

    loss_fn.projection.data = torch.from_numpy(projection)

    def normalize(matr):
        return matr / np.sqrt((matr ** 2).sum(axis=1))[:, np.newaxis]

    normalized_features = normalize(features)
    normalized_projection = normalize(projection)

    cosine = normalized_features @ normalized_projection.T
    phi = cosine - m

    mask = np.array([[1, 0, 0], [0, 0, 1]], dtype="l")
    feats = (mask * phi + (1.0 - mask) * cosine) * s

    expected_loss = 1.3651
    actual = loss_fn(torch.from_numpy(features), torch.LongTensor(target))
    assert abs(expected_loss - actual.item()) < 1e-5
