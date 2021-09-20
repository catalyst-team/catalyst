# flake8: noqa
import os

from pytest import mark

from torch import nn, optim
from torch.utils.data import DataLoader

from catalyst import dl
from catalyst.contrib.datasets import MNIST
from catalyst.data import (
    BalanceClassSampler,
    BatchBalanceClassSampler,
    BatchPrefetchLoaderWrapper,
    ToTensor,
)
from catalyst.settings import IS_CUDA_AVAILABLE


def test_balance_class_sampler():
    train_data = MNIST(os.getcwd(), train=True, download=True, transform=ToTensor())
    train_labels = train_data.targets.cpu().numpy().tolist()
    train_sampler = BalanceClassSampler(train_labels, mode=5000)
    valid_data = MNIST(os.getcwd(), train=False, download=True, transform=ToTensor())

    loaders = {
        "train": DataLoader(train_data, sampler=train_sampler, batch_size=32),
        "valid": DataLoader(valid_data, batch_size=32),
    }

    model = nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, 10))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.02)

    runner = dl.SupervisedRunner()
    runner.train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        loaders=loaders,
        num_epochs=1,
        logdir="./logs",
        valid_loader="valid",
        valid_metric="loss",
        minimize_valid_metric=True,
        verbose=True,
    )


def test_batch_balance_class_sampler():
    train_data = MNIST(os.getcwd(), train=True, download=True, transform=ToTensor())
    train_labels = train_data.targets.cpu().numpy().tolist()
    train_sampler = BatchBalanceClassSampler(train_labels, num_classes=10, num_samples=4)
    valid_data = MNIST(os.getcwd(), train=False, download=True, transform=ToTensor())

    loaders = {
        "train": DataLoader(train_data, batch_sampler=train_sampler),
        "valid": DataLoader(valid_data, batch_size=32),
    }

    model = nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, 10))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.02)

    runner = dl.SupervisedRunner()
    runner.train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        loaders=loaders,
        num_epochs=1,
        logdir="./logs",
        valid_loader="valid",
        valid_metric="loss",
        minimize_valid_metric=True,
        verbose=True,
    )


@mark.skipif(not IS_CUDA_AVAILABLE, reason="CUDA device is not available")
def test_balance_class_sampler_with_prefetch():
    train_data = MNIST(os.getcwd(), train=True, download=True, transform=ToTensor())
    train_labels = train_data.targets.cpu().numpy().tolist()
    train_sampler = BalanceClassSampler(train_labels, mode=5000)
    valid_data = MNIST(os.getcwd(), train=False, download=True, transform=ToTensor())

    loaders = {
        "train": DataLoader(train_data, sampler=train_sampler, batch_size=32),
        "valid": DataLoader(valid_data, batch_size=32),
    }
    loaders = {k: BatchPrefetchLoaderWrapper(v) for k, v in loaders.items()}

    model = nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, 10))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.02)

    runner = dl.SupervisedRunner()
    runner.train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        loaders=loaders,
        num_epochs=1,
        logdir="./logs",
        valid_loader="valid",
        valid_metric="loss",
        minimize_valid_metric=True,
        verbose=True,
    )


@mark.skipif(not IS_CUDA_AVAILABLE, reason="CUDA device is not available")
def test_batch_balance_class_sampler_with_prefetch():
    train_data = MNIST(os.getcwd(), train=True, download=True, transform=ToTensor())
    train_labels = train_data.targets.cpu().numpy().tolist()
    train_sampler = BatchBalanceClassSampler(train_labels, num_classes=10, num_samples=4)
    valid_data = MNIST(os.getcwd(), train=False, download=True, transform=ToTensor())

    loaders = {
        "train": DataLoader(train_data, batch_sampler=train_sampler),
        "valid": DataLoader(valid_data, batch_size=32),
    }
    loaders = {k: BatchPrefetchLoaderWrapper(v) for k, v in loaders.items()}

    model = nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, 10))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.02)

    runner = dl.SupervisedRunner()
    runner.train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        loaders=loaders,
        num_epochs=1,
        logdir="./logs",
        valid_loader="valid",
        valid_metric="loss",
        minimize_valid_metric=True,
        verbose=True,
    )
