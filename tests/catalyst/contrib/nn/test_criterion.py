# flake8: noqa

import numpy as np
import pytest

import torch

from catalyst.contrib.nn import criterion as module
from catalyst.contrib.nn.criterion import (
    BarlowTwinsLoss,
    CircleLoss,
    NTXentLoss,
    SupervisedContrastiveLoss,
    TripletMarginLossWithSampler,
)
from catalyst.data import AllTripletsSampler


def test_criterion_init():
    """@TODO: Docs. Contribution is welcome."""
    for module_class in module.__dict__.values():
        if isinstance(module_class, type):
            if module_class == CircleLoss:
                instance = module_class(margin=0.25, gamma=256)
            elif module_class == TripletMarginLossWithSampler:
                instance = module_class(margin=1.0, sampler_inbatch=AllTripletsSampler())
            elif module_class == BarlowTwinsLoss:
                instance = module_class(offdiag_lambda=1, eps=1e-12)
            elif module_class == NTXentLoss:
                instance = module_class(tau=0.1)
            elif module_class == SupervisedContrastiveLoss:
                instance = module_class(tau=0.1, reduction="mean", pos_aggregation="in")
            else:
                # @TODO: very dirty trick
                try:
                    instance = module_class()
                except:
                    print(module_class)
                    instance = 1
            assert instance is not None


def test_bpr_loss():
    """Testing for Bayesian Personalized Ranking"""
    from catalyst.contrib.nn.criterion.recsys import BPRLoss

    loss = BPRLoss()

    rand = torch.rand(1000, dtype=torch.float)
    assert float(loss.forward(rand, rand)) == pytest.approx(0.6931, 0.001)  # log of 0.5

    pos, neg = torch.Tensor([1000, 1000, 1000, 1000]), torch.Tensor([0, 0, 0, 0])
    assert float(loss.forward(pos, neg)) == pytest.approx(0, 0.001)  # log of 1

    neg, pos = torch.Tensor([1000, 1000, 1000, 1000]), torch.Tensor([0, 0, 0, 0])
    log_gamma = float(torch.log(torch.Tensor([loss.gamma])))
    assert float(loss.forward(pos, neg)) == pytest.approx(-log_gamma, 0.001)  # log of 0 with gamma


def test_warp_loss():
    from catalyst.contrib.nn.criterion.recsys import WARPLoss

    loss = WARPLoss(max_num_trials=1)

    outputs = torch.Tensor([[0, 0, 0, 1]])
    targets = torch.Tensor([[0, 0, 0, 1]])
    assert float(loss.forward(outputs, targets)) == pytest.approx(0, 0.001)

    x2_outputs = torch.stack((outputs.squeeze(0), outputs.squeeze(0)))
    x2_targets = torch.stack((targets.squeeze(0), targets.squeeze(0)))
    assert float(loss.forward(x2_outputs, x2_targets)) == pytest.approx(0, 0.001)

    outputs = torch.Tensor([[0, 0, 0, 0]])
    targets = torch.Tensor([[0, 0, 0, 1]])
    assert float(loss.forward(outputs, targets)) == pytest.approx(1.0986, 0.001)

    x2_outputs = torch.stack((outputs.squeeze(0), outputs.squeeze(0)))
    x2_target = torch.stack((targets.squeeze(0), targets.squeeze(0)))
    assert float(loss.forward(x2_outputs, x2_target)) == pytest.approx(2 * 1.0986, 0.001)

    outputs = torch.Tensor([[0.5, 0.5, 0.5, 1]])
    targets = torch.Tensor([[0, 0, 0, 1]])
    assert float(loss.forward(outputs, targets)) == pytest.approx(0.5493, 0.001)

    outputs = torch.Tensor([[0.5, 0, 0.5, 1]])
    targets = torch.Tensor([[0, 0, 0, 1]])
    loss_value = float(loss.forward(outputs, targets))
    assert loss_value == pytest.approx(0.5493, 0.001) or loss_value == pytest.approx(0, 0.001)


def test_logistic_loss():
    from catalyst.contrib.nn.criterion.recsys import LogisticLoss

    loss = LogisticLoss()

    rand = torch.rand(1000)
    assert float(loss.forward(rand, rand)) == pytest.approx(1, 0.001)  # neg relu of 1

    pos, neg = torch.Tensor([1000, 1000, 1000, 1000]), torch.Tensor([0, 0, 0, 0])
    assert float(loss.forward(pos, neg)) == pytest.approx(0.5, 0.001)  # relu of large negative

    neg, pos = torch.Tensor([1000, 1000, 1000, 1000]), torch.Tensor([0, 0, 0, 0])
    assert float(loss.forward(pos, neg)) == pytest.approx(1.5, 0.001)  # nerelu of large positive


def test_hinge_loss():
    from catalyst.contrib.nn.criterion.recsys import HingeLoss

    loss = HingeLoss()

    rand = torch.rand(1000)
    assert float(loss.forward(rand, rand)) == pytest.approx(1, 0.001)  # relu of 0

    pos, neg = torch.Tensor([1, 1, 1, 1]), torch.Tensor([0, 0, 0, 0])
    assert float(loss.forward(pos, neg)) == pytest.approx(0, 0.001)  # relu of 1

    pos, neg = torch.Tensor([1000, 1000, 1000, 1000]), torch.Tensor([0, 0, 0, 0])
    assert float(loss.forward(pos, neg)) == pytest.approx(0, 0.001)  # relu of large negative

    neg, pos = torch.Tensor([1000, 1000, 1000, 1000]), torch.Tensor([0, 0, 0, 0])
    assert float(loss.forward(pos, neg)) == pytest.approx(1001, 0.001)  # nerelu of large positive


def test_adaptive_hinge_loss():
    from catalyst.contrib.nn.criterion.recsys import AdaptiveHingeLoss

    loss = AdaptiveHingeLoss()

    rand = torch.rand(1000)
    ones = torch.ones(1000)
    assert float(loss.forward(rand, rand.unsqueeze(0))) == pytest.approx(1, 0.001)  # relu of 0
    assert float(loss.forward(rand, torch.stack((rand, rand)))) == pytest.approx(1, 0.001)
    assert float(loss.forward(ones, torch.stack((rand, ones)))) == pytest.approx(1, 0.001)

    pos, neg = torch.Tensor([1, 1, 1, 1]), torch.Tensor([0, 0, 0, 0]).unsqueeze(0)
    assert float(loss.forward(pos, neg)) == pytest.approx(0, 0.001)  # relu of 1

    pos, neg = torch.Tensor([1000, 1000, 1000, 1000]), torch.Tensor([0, 0, 0, 0]).unsqueeze(0)
    assert float(loss.forward(pos, neg)) == pytest.approx(0, 0.001)  # relu of large negative

    pos, neg = torch.Tensor([0, 0, 0, 0]), torch.Tensor([1000, 1000, 1000, 1000]).unsqueeze(0)
    assert float(loss.forward(pos, neg)) == pytest.approx(1001, 0.001)  # nerelu of large positive


def test_roc_star_loss():
    from catalyst.contrib.nn.criterion.recsys import RocStarLoss

    params = dict(sample_size=5, sample_size_gamma=5, update_gamma_each=1)
    const_history = torch.Tensor([[0], [1], [0], [0], [1], [1], [0], [1], [0], [1]])  # rand seq

    outputs = torch.Tensor([[0], [1], [0], [1], [0]])
    targets = torch.Tensor([[1], [0], [1], [0], [1]])

    loss = RocStarLoss(**params)
    loss.outputs_history = const_history
    loss.targets_history = const_history
    assert float(loss.forward(outputs, outputs)) == pytest.approx(0, 0.001)

    loss.__init__(**params)
    loss.outputs_history = const_history
    loss.targets_history = const_history
    assert float(loss.forward(targets, targets)) == pytest.approx(0, 0.001)

    loss.__init__(**params)
    loss.outputs_history = const_history
    loss.targets_history = const_history
    assert float(loss.forward(outputs, targets)) == pytest.approx(2, 0.001)


@pytest.mark.parametrize(
    "embeddings_left,embeddings_right,offdiag_lambda,eps,true_value",
    (
        (
            torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            1,
            1e-12,
            1,
        ),
        (
            torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            0,
            1e-12,
            0.5,
        ),
        (
            torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            2,
            1e-12,
            1.5,
        ),
        (
            torch.tensor(
                [
                    [-0.31887834],
                    [1.3980029],
                    [0.30775256],
                    [0.29397671],
                    [-1.47968253],
                    [-0.72796992],
                    [-0.30937596],
                    [1.16363952],
                    [-2.15524895],
                    [-0.0440765],
                ]
            ),
            torch.tensor(
                [
                    [-0.31887834],
                    [1.3980029],
                    [0.30775256],
                    [0.29397671],
                    [-1.47968253],
                    [-0.72796992],
                    [-0.30937596],
                    [1.16363952],
                    [-2.15524895],
                    [-0.0440765],
                ]
            ),
            1,
            1e-12,
            0.01,
        ),
    ),
)
def test_barlow_twins_loss(
    embeddings_left: torch.Tensor,
    embeddings_right: torch.Tensor,
    offdiag_lambda: float,
    eps: float,
    true_value: float,
):
    """
    Test Barlow Twins loss

    Args:
        embeddings_left: left objects embeddings [batch_size, features_dim]
        embeddings_right: right objects embeddings [batch_size, features_dim]
        offdiag_lambda: trade off parametr
        eps: zero varience handler (var + eps)
        true_value: expected loss value
    """
    value = BarlowTwinsLoss(offdiag_lambda=offdiag_lambda, eps=eps)(
        embeddings_left, embeddings_right
    ).item()
    assert np.isclose(value, true_value)


@pytest.mark.parametrize(
    "embeddings_left,embeddings_right,tau,true_value",
    (
        (
            torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
            torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
            1,
            0.90483244155,
        ),
    ),
)
def test_ntxent_loss(
    embeddings_left: torch.Tensor, embeddings_right: torch.Tensor, tau: float, true_value: float
):
    """
    Test NTXent Loss

    Args:
        embeddings_left: left objects embeddings [batch_size, features_dim]
        embeddings_right: right objects embeddings [batch_size, features_dim]
        tau: temperature
        true_value: expected loss value
    """
    value = NTXentLoss(tau=tau)(embeddings_left, embeddings_right).item()
    assert np.isclose(value, true_value)


@pytest.mark.parametrize(
    "features,targets,tau,pos_aggregation,true_value",
    (
        (
            torch.tensor(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ]
            ),
            torch.tensor([1, 2, 3, 1, 2, 3]),
            1,
            "in",
            0.90483244155,
        ),
        (
            torch.tensor(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ]
            ),
            torch.tensor([1, 2, 3, 1, 2, 3]),
            1,
            "out",
            0.90483244155,
        ),
    ),
)
def test_supervised_contrastive_loss(
    features: torch.Tensor,
    targets: torch.Tensor,
    tau: float,
    pos_aggregation: str,
    true_value: float,
):
    """
    Test supervised contrastive loss

    Args:
        features: features of objects
        targets: targets of objects
        pos_aggregation: aggeragation of positive objects
        tau: temperature
        true_value: expected loss value
    """
    value = SupervisedContrastiveLoss(tau=tau, pos_aggregation=pos_aggregation)(
        features, targets
    ).item()
    assert np.isclose(value, true_value)
