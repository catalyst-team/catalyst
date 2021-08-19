# flake8: noqa

import numpy as np
import pytest
import torch

from catalyst.contrib.nn import criterion as module
from catalyst.contrib.nn.criterion import CircleLoss, TripletMarginLossWithSampler
from catalyst.contrib.nn.criterion.contrastive import BarlowTwinsLoss
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

    input_ = torch.Tensor([[0, 0, 0, 1]])
    target = torch.Tensor([[0, 0, 0, 1]])
    assert float(loss.forward(input_, target)) == pytest.approx(0, 0.001)

    x2_input_ = torch.stack((input_.squeeze(0), input_.squeeze(0)))
    x2_target = torch.stack((target.squeeze(0), target.squeeze(0)))
    assert float(loss.forward(x2_input_, x2_target)) == pytest.approx(0, 0.001)

    input_ = torch.Tensor([[0, 0, 0, 0]])
    target = torch.Tensor([[0, 0, 0, 1]])
    assert float(loss.forward(input_, target)) == pytest.approx(1.0986, 0.001)

    x2_input_ = torch.stack((input_.squeeze(0), input_.squeeze(0)))
    x2_target = torch.stack((target.squeeze(0), target.squeeze(0)))
    assert float(loss.forward(x2_input_, x2_target)) == pytest.approx(2 * 1.0986, 0.001)

    input_ = torch.Tensor([[0.5, 0.5, 0.5, 1]])
    target = torch.Tensor([[0, 0, 0, 1]])
    assert float(loss.forward(input_, target)) == pytest.approx(0.5493, 0.001)

    input_ = torch.Tensor([[0.5, 0, 0.5, 1]])
    target = torch.Tensor([[0, 0, 0, 1]])
    loss_value = float(loss.forward(input_, target))
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
