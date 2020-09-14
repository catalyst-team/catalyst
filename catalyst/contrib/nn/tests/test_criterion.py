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

    loss_fn = CosFaceLoss(emb_size, n_classes, s, m, reduction="none")
    loss_fn.projection.data = torch.from_numpy(projection)

    def normalize(matr):
        return (
            matr / np.sqrt((matr ** 2).sum(axis=1))[:, np.newaxis]
        )  # for each row

    def softmax(x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(1)[:, np.newaxis]  # for each row

    def cross_entropy(preds, targs, axis=None):
        print(softmax(preds))
        return -(targs * np.log(softmax(preds))).sum(axis)

    normalized_features = normalize(features)  # 2x4
    normalized_projection = normalize(projection)  # 3x4

    cosine = normalized_features @ normalized_projection.T  # 2x4 * 4x3 = 2x3
    phi = cosine - m  # 2x3

    mask = np.array([[1, 0, 0], [0, 0, 1]], dtype="l")  # one_hot(target)
    feats = (mask * phi + (1.0 - mask) * cosine) * s  # 2x3

    expected_loss = cross_entropy(feats, mask, 1)
    actual = (
        loss_fn(torch.from_numpy(features), torch.LongTensor(target))
        .detach()
        .numpy()
    )
    assert np.allclose(expected_loss, actual)

    loss_fn = CosFaceLoss(emb_size, n_classes, s, m, reduction="mean")
    loss_fn.projection.data = torch.from_numpy(projection)

    expected_loss = cross_entropy(feats, mask, 1)
    actual = (
        loss_fn(torch.from_numpy(features), torch.LongTensor(target))
        .detach()
        .numpy()
    )
    assert np.isclose(expected_loss.mean(), actual)

    loss_fn = CosFaceLoss(emb_size, n_classes, s, m, reduction="sum")
    loss_fn.projection.data = torch.from_numpy(projection)

    expected_loss = cross_entropy(feats, mask, 1)
    actual = (
        loss_fn(torch.from_numpy(features), torch.LongTensor(target))
        .detach()
        .numpy()
    )
    assert np.isclose(expected_loss.sum(), actual)