# flake8: noqa

import os

from pytest import mark

import torch
from torch import nn
from torch.utils.data import DataLoader

from catalyst import dl
from catalyst.contrib.datasets import MNIST
from tests import DATA_ROOT


def _is_profile_available():
    try:
        from torch import profiler  # noqa: F401

        return True
    except ImportError:
        return False


def train_experiment():
    loaders = {
        "train": DataLoader(
            MNIST(DATA_ROOT, train=False),
            batch_size=32,
        ),
        "valid": DataLoader(
            MNIST(DATA_ROOT, train=False),
            batch_size=32,
        ),
    }
    model = nn.Sequential(
        nn.Flatten(), nn.Linear(784, 512), nn.ReLU(), nn.Linear(512, 10)
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    runner = dl.SupervisedRunner()
    runner.train(
        model=model,
        loaders=loaders,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=5,
        logdir="./logs",
        profile=True,
    )


@mark.skipif(not _is_profile_available(), reason="Torch profiler is not available")
def test_profiler():
    train_experiment()
    assert (
        os.path.isdir("./logs/tb_profile")
        and not len(os.listdir("./logs/tb_profile")) == 0
    )
