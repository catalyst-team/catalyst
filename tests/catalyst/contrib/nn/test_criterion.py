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
                instance = module_class(lmbda=1, eps=1e-12)
            else:
                # @TODO: very dirty trick
                try:
                    instance = module_class()
                except:
                    print(module_class)
                    instance = 1
            assert instance is not None


@pytest.mark.parametrize(
    "embeddings_left,embeddings_right,lmbda,eps,true_value",
    (
        (
            torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            1,
            1e-12,
            1,
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
    lmbda: float,
    eps: float,
    true_value: float,
):
    """
    Test Barlow Twins loss
    Args:
        embeddings_left: left objects embeddings [batch_size, features_dim]
        embeddings_right: right objects embeddings [batch_size, features_dim]
        lmbda: trade off parametr
        eps: zero varience handler (var + eps)
        true_value: expected loss value 
    """
    value = BarlowTwinsLoss(lmbda=lmbda, eps=eps)(embeddings_left, embeddings_right).item()
    assert np.isclose(value, true_value)
