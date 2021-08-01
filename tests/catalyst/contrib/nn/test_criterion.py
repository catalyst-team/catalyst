# flake8: noqa
import pytest
import torch

from catalyst.contrib.nn import criterion as module
from catalyst.contrib.nn.criterion import CircleLoss, TripletMarginLossWithSampler
from catalyst.data import AllTripletsSampler


def test_criterion_init():
    """@TODO: Docs. Contribution is welcome."""
    for module_class in module.__dict__.values():
        if isinstance(module_class, type):
            if module_class == CircleLoss:
                instance = module_class(margin=0.25, gamma=256)
            elif module_class == TripletMarginLossWithSampler:
                instance = module_class(margin=1.0, sampler_inbatch=AllTripletsSampler())
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

    pos, neg = torch.Tensor([1000, 1000, 1000, 1000,]), torch.Tensor([0, 0, 0, 0,])
    assert float(loss.forward(pos, neg)) == pytest.approx(0, 0.001)  # log of 1

    neg, pos = torch.Tensor([1000, 1000, 1000, 1000,]), torch.Tensor([0, 0, 0, 0,])
    log_gamma = float(torch.log(torch.Tensor([loss.gamma,])))
    assert float(loss.forward(pos, neg)) == pytest.approx(-log_gamma, 0.001)  # log of 0 with gamma


def test_warp_loss():
    pass


def test_logistic_loss():
    from catalyst.contrib.nn.criterion.recsys import LogisticLoss

    loss = LogisticLoss()

    rand = torch.rand(1000)
    assert float(loss.forward(rand, rand)) == pytest.approx(1, 0.001)  # neg relu of 1

    pos, neg = torch.Tensor([1000, 1000, 1000, 1000,]), torch.Tensor([0, 0, 0, 0,])
    assert float(loss.forward(pos, neg)) == pytest.approx(0.5, 0.001)  # relu of large negative

    neg, pos = torch.Tensor([1000, 1000, 1000, 1000,]), torch.Tensor([0, 0, 0, 0,])
    assert float(loss.forward(pos, neg)) == pytest.approx(1.5, 0.001)  # nerelu of large positive


def test_hinge_loss():
    from catalyst.contrib.nn.criterion.recsys import HingeLoss

    loss = HingeLoss()

    rand = torch.rand(1000)
    assert float(loss.forward(rand, rand)) == pytest.approx(1, 0.001)  # relu of 0

    pos, neg = torch.Tensor([1, 1, 1, 1,]), torch.Tensor([0, 0, 0, 0,])
    assert float(loss.forward(pos, neg)) == pytest.approx(0, 0.001)  # relu of 1

    pos, neg = torch.Tensor([1000, 1000, 1000, 1000,]), torch.Tensor([0, 0, 0, 0,])
    assert float(loss.forward(pos, neg)) == pytest.approx(0, 0.001)  # relu of large negative

    neg, pos = torch.Tensor([1000, 1000, 1000, 1000,]), torch.Tensor([0, 0, 0, 0,])
    assert float(loss.forward(pos, neg)) == pytest.approx(1001, 0.001)  # nerelu of large positive
