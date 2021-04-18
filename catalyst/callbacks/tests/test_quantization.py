# flake8: noqa
import os

import torch
from torch import nn
from torch.utils.data import DataLoader

from catalyst import dl
from catalyst.contrib.datasets import MNIST
from catalyst.contrib.nn.modules import Flatten
from catalyst.data.transforms import ToTensor


def test_pruning_callback() -> None:
    """Quantize model"""
    loaders = {
        "train": DataLoader(
            MNIST(os.getcwd(), train=False, download=True, transform=ToTensor()), batch_size=32,
        ),
        "valid": DataLoader(
            MNIST(os.getcwd(), train=False, download=True, transform=ToTensor()), batch_size=32,
        ),
    }
    model = nn.Sequential(Flatten(), nn.Linear(784, 512), nn.ReLU(), nn.Linear(512, 10))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    runner = dl.SupervisedRunner()
    runner.train(
        model=model,
        callbacks=[dl.QuantizationCallback(logdir="./logs")],
        loaders=loaders,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=1,
        logdir="./logs",
        check=True,
    )
    assert os.path.isfile("./logs/quantized.pth")
