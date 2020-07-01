# flake8: noqa
import copy
from io import StringIO
import os
import re
import shutil
import sys

import pytest

import torch
from torch.utils.data import DataLoader, TensorDataset

from catalyst.dl import (
    Callback,
    CallbackOrder,
    CriterionCallback,
    PeriodicLoaderCallback,
    SupervisedRunner,
)


def test_multiple_stages_with_magic_callback():
    # NOTE: before first validation epoch
    # all checkpoints will be compared according
    # to a metric on a test dataset and
    # checkpoints will be overwritten according
    # to this value
    class BestStateCheckerCallback(Callback):
        def __init__(self):
            super().__init__(CallbackOrder.External)
            self.valid_loader = None
            self._after_first_validation = False

        def on_stage_start(self, runner: "IRunner") -> None:
            self.valid_loader = copy.copy(runner.valid_loader)

        def on_epoch_end(self, runner: "IRunner") -> None:
            if (
                self.valid_loader not in runner.loaders
                and runner.epoch > 1
                and self._after_first_validation
            ):
                assert (
                    not runner.is_best_valid
                ), f"Epochs (epoch={runner.epoch}) without valid loader can't be best!"
            else:
                assert runner.valid_metrics[runner.main_metric] is not None
            if self.valid_loader in runner.loaders:
                self._after_first_validation = True

    # experiment_setup
    logdir = "./logs/periodic_loader"

    # data
    num_samples, num_features = int(1e4), int(1e1)
    X = torch.rand(num_samples, num_features)
    y = torch.randint(0, 5, size=[num_samples])
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=32, num_workers=1)
    loaders = {
        "train": loader,
        "valid": loader,
    }

    # model, criterion, optimizer, scheduler
    model = torch.nn.Linear(num_features, 5)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    runner = SupervisedRunner()

    # first stage
    runner.train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        loaders=loaders,
        logdir=logdir,
        num_epochs=5,
        verbose=False,
        callbacks=[
            PeriodicLoaderCallback(valid=2),
            BestStateCheckerCallback(),
        ],
    )

    # second stage
    runner.train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        loaders=loaders,
        logdir=logdir,
        num_epochs=6,
        verbose=False,
        callbacks=[
            PeriodicLoaderCallback(valid=3),
            BestStateCheckerCallback(),
        ],
    )

    # third stage
    runner.train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        loaders=loaders,
        logdir=logdir,
        num_epochs=6,
        verbose=False,
        callbacks=[
            PeriodicLoaderCallback(valid=4),
            BestStateCheckerCallback(),
        ],
    )

    shutil.rmtree(logdir, ignore_errors=True)


def test_validation_with_period_3():
    old_stdout = sys.stdout
    sys.stdout = str_stdout = StringIO()

    # experiment_setup
    logdir = "./logs/periodic_loader"
    checkpoint = logdir + "/checkpoints"
    logfile = checkpoint + "/_metrics.json"

    # data
    num_samples, num_features = int(1e4), int(1e1)
    X = torch.rand(num_samples, num_features)
    y = torch.randint(0, 5, size=[num_samples])
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=32, num_workers=1)
    loaders = {
        "train": loader,
        "valid": loader,
    }

    # model, criterion, optimizer, scheduler
    model = torch.nn.Linear(num_features, 5)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    runner = SupervisedRunner()

    # first stage
    runner.train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        loaders=loaders,
        logdir=logdir,
        num_epochs=10,
        verbose=False,
        callbacks=[PeriodicLoaderCallback(valid=3)],
    )

    sys.stdout = old_stdout
    exp_output = str_stdout.getvalue()

    assert len(re.findall(r"\(train\)", exp_output)) == 10
    assert len(re.findall(r"\(valid\)", exp_output)) == 3
    assert len(re.findall(r".*/train\.\d\.pth", exp_output)) == 1

    assert os.path.isfile(logfile)
    assert os.path.isfile(checkpoint + "/train.9.pth")
    assert os.path.isfile(checkpoint + "/best.pth")
    assert os.path.isfile(checkpoint + "/best_full.pth")
    assert os.path.isfile(checkpoint + "/last.pth")
    assert os.path.isfile(checkpoint + "/last_full.pth")

    shutil.rmtree(logdir, ignore_errors=True)


def test_validation_with_period_0():
    old_stdout = sys.stdout
    sys.stdout = str_stdout = StringIO()

    # experiment_setup
    logdir = "./logs/periodic_loader"
    checkpoint = logdir + "/checkpoints"
    logfile = checkpoint + "/_metrics.json"

    # data
    num_samples, num_features = int(1e4), int(1e1)
    X = torch.rand(num_samples, num_features)
    y = torch.randint(0, 5, size=[num_samples])
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=32, num_workers=1)
    loaders = {
        "train": loader,
        "valid": loader,
    }

    # model, criterion, optimizer, scheduler
    model = torch.nn.Linear(num_features, 5)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    runner = SupervisedRunner()

    # first stage
    runner.train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        loaders=loaders,
        logdir=logdir,
        num_epochs=5,
        verbose=False,
        callbacks=[PeriodicLoaderCallback(valid=0)],
    )

    sys.stdout = old_stdout
    exp_output = str_stdout.getvalue()

    assert len(re.findall(r"\(train\)", exp_output)) == 5
    assert len(re.findall(r"\(valid\)", exp_output)) == 0
    assert len(re.findall(r".*/train\.\d\.pth", exp_output)) == 1

    assert os.path.isfile(logfile)
    assert os.path.isfile(checkpoint + "/train.5.pth")
    assert os.path.isfile(checkpoint + "/best.pth")
    assert os.path.isfile(checkpoint + "/best_full.pth")
    assert os.path.isfile(checkpoint + "/last.pth")
    assert os.path.isfile(checkpoint + "/last_full.pth")

    # shutil.rmtree(logdir, ignore_errors=True)


def test_multiple_loaders():
    old_stdout = sys.stdout
    sys.stdout = str_stdout = StringIO()

    # experiment_setup
    logdir = "./logs/periodic_loader"
    checkpoint = logdir + "/checkpoints"
    logfile = checkpoint + "/_metrics.json"

    # data
    num_samples, num_features = int(1e4), int(1e1)
    X = torch.rand(num_samples, num_features)
    y = torch.randint(0, 5, size=[num_samples])
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=32, num_workers=1)
    loaders = {
        "train": loader,
        "train_additional": loader,
        "valid": loader,
        "valid_additional": loader,
    }

    # model, criterion, optimizer, scheduler
    model = torch.nn.Linear(num_features, 5)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    runner = SupervisedRunner()

    # first stage
    runner.train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        loaders=loaders,
        logdir=logdir,
        num_epochs=10,
        verbose=False,
        callbacks=[
            PeriodicLoaderCallback(
                train_additional=2, valid=3, valid_additional=0
            )
        ],
    )

    sys.stdout = old_stdout
    exp_output = str_stdout.getvalue()

    assert len(re.findall(r"\(train\)", exp_output)) == 10
    assert len(re.findall(r"\(train_additional\)", exp_output)) == 5
    assert len(re.findall(r"\(valid\)", exp_output)) == 3
    assert len(re.findall(r"\(valid_additional\)", exp_output)) == 0
    assert len(re.findall(r".*/train\.\d\.pth", exp_output)) == 1

    assert os.path.isfile(logfile)
    assert os.path.isfile(checkpoint + "/train.9.pth")
    assert os.path.isfile(checkpoint + "/best.pth")
    assert os.path.isfile(checkpoint + "/best_full.pth")
    assert os.path.isfile(checkpoint + "/last.pth")
    assert os.path.isfile(checkpoint + "/last_full.pth")

    shutil.rmtree(logdir, ignore_errors=True)


def test_multiple_loaders_and_multiple_stages():
    old_stdout = sys.stdout
    sys.stdout = str_stdout = StringIO()

    # experiment_setup
    logdir = "./logs/periodic_loader"
    checkpoint = logdir + "/checkpoints"
    logfile = checkpoint + "/_metrics.json"

    # data
    num_samples, num_features = int(1e4), int(1e1)
    X = torch.rand(num_samples, num_features)
    y = torch.randint(0, 5, size=[num_samples])
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=32, num_workers=1)
    loaders = {
        "train": loader,
        "train_additional": loader,
        "valid": loader,
        "valid_additional": loader,
    }

    # model, criterion, optimizer, scheduler
    model = torch.nn.Linear(num_features, 5)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    runner = SupervisedRunner()

    # first stage
    runner.train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        loaders=loaders,
        logdir=logdir,
        num_epochs=5,
        verbose=False,
        callbacks=[
            PeriodicLoaderCallback(
                train_additional=2, valid=3, valid_additional=0
            )
        ],
    )

    # second stage
    runner.train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        loaders=loaders,
        logdir=logdir,
        num_epochs=10,
        verbose=False,
        callbacks=[
            PeriodicLoaderCallback(
                train_additional=2, valid=3, valid_additional=0
            )
        ],
    )

    sys.stdout = old_stdout
    exp_output = str_stdout.getvalue()

    assert len(re.findall(r"\(train\)", exp_output)) == 15
    assert len(re.findall(r"\(train_additional\)", exp_output)) == 7
    assert len(re.findall(r"\(valid\)", exp_output)) == 4
    assert len(re.findall(r"\(valid_additional\)", exp_output)) == 0
    assert len(re.findall(r".*/train\.\d\.pth", exp_output)) == 2

    assert os.path.isfile(logfile)
    assert os.path.isfile(checkpoint + "/train.9.pth")
    assert os.path.isfile(checkpoint + "/best.pth")
    assert os.path.isfile(checkpoint + "/best_full.pth")
    assert os.path.isfile(checkpoint + "/last.pth")
    assert os.path.isfile(checkpoint + "/last_full.pth")

    shutil.rmtree(logdir, ignore_errors=True)


def test_no_loaders_epoch():
    old_stdout = sys.stdout
    sys.stdout = str_stdout = StringIO()

    # experiment_setup
    logdir = "./logs/periodic_loader"
    checkpoint = logdir + "/checkpoints"
    logfile = checkpoint + "/_metrics.json"

    # data
    num_samples, num_features = int(1e4), int(1e1)
    X = torch.rand(num_samples, num_features)
    y = torch.randint(0, 5, size=[num_samples])
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=32, num_workers=1)
    loaders = {
        "train": loader,
        "train_additional": loader,
        "valid": loader,
        "valid_additional": loader,
    }

    # model, criterion, optimizer, scheduler
    model = torch.nn.Linear(num_features, 5)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    runner = SupervisedRunner()

    with pytest.raises(ValueError):
        runner.train(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            loaders=loaders,
            logdir=logdir,
            num_epochs=10,
            verbose=False,
            callbacks=[
                PeriodicLoaderCallback(
                    train=2, train_additional=2, valid=3, valid_additional=0
                )
            ],
        )

    sys.stdout = old_stdout
    exp_output = str_stdout.getvalue()

    shutil.rmtree(logdir, ignore_errors=True)


def test_wrong_period_type():
    old_stdout = sys.stdout
    sys.stdout = str_stdout = StringIO()

    # experiment_setup
    logdir = "./logs/periodic_loader"
    checkpoint = logdir + "/checkpoints"
    logfile = checkpoint + "/_metrics.json"

    # data
    num_samples, num_features = int(1e4), int(1e1)
    X = torch.rand(num_samples, num_features)
    y = torch.randint(0, 5, size=[num_samples])
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=32, num_workers=1)
    loaders = {
        "train": loader,
        "train_additional": loader,
        "valid": loader,
        "valid_additional": loader,
    }

    # model, criterion, optimizer, scheduler
    model = torch.nn.Linear(num_features, 5)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    runner = SupervisedRunner()

    with pytest.raises(TypeError):
        runner.train(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            loaders=loaders,
            logdir=logdir,
            num_epochs=10,
            verbose=False,
            callbacks=[
                PeriodicLoaderCallback(
                    train_additional=[],
                    train_not_exists=2,
                    valid=3,
                    valid_additional=0,
                    valid_not_exist=1,
                )
            ],
        )

    sys.stdout = old_stdout
    exp_output = str_stdout.getvalue()

    shutil.rmtree(logdir, ignore_errors=True)


def test_ignoring_unknown_loaders():
    old_stdout = sys.stdout
    sys.stdout = str_stdout = StringIO()

    # experiment_setup
    logdir = "./logs/periodic_loader"
    checkpoint = logdir + "/checkpoints"
    logfile = checkpoint + "/_metrics.json"

    # data
    num_samples, num_features = int(1e4), int(1e1)
    X = torch.rand(num_samples, num_features)
    y = torch.randint(0, 5, size=[num_samples])
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=32, num_workers=1)
    loaders = {
        "train": loader,
        "train_additional": loader,
        "valid": loader,
        "valid_additional": loader,
    }

    # model, criterion, optimizer, scheduler
    model = torch.nn.Linear(num_features, 5)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    runner = SupervisedRunner()

    # first stage
    runner.train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        loaders=loaders,
        logdir=logdir,
        num_epochs=10,
        verbose=False,
        callbacks=[
            PeriodicLoaderCallback(
                train_additional=2,
                train_not_exists=2,
                valid=3,
                valid_additional=0,
                valid_not_exist=1,
            )
        ],
    )

    sys.stdout = old_stdout
    exp_output = str_stdout.getvalue()

    assert len(re.findall(r"\(train\)", exp_output)) == 10
    assert len(re.findall(r"\(train_additional\)", exp_output)) == 5
    assert len(re.findall(r"\(train_not_exists\)", exp_output)) == 0
    assert len(re.findall(r"\(valid\)", exp_output)) == 3
    assert len(re.findall(r"\(valid_additional\)", exp_output)) == 0
    assert len(re.findall(r"\(valid_not_exist\)", exp_output)) == 0
    assert len(re.findall(r".*/train\.\d\.pth", exp_output)) == 1

    assert os.path.isfile(logfile)
    assert os.path.isfile(checkpoint + "/train.9.pth")
    assert os.path.isfile(checkpoint + "/best.pth")
    assert os.path.isfile(checkpoint + "/best_full.pth")
    assert os.path.isfile(checkpoint + "/last.pth")
    assert os.path.isfile(checkpoint + "/last_full.pth")

    shutil.rmtree(logdir, ignore_errors=True)


def test_loading_best_state_at_end():
    old_stdout = sys.stdout
    sys.stdout = str_stdout = StringIO()

    # experiment_setup
    logdir = "./logs/periodic_loader"
    checkpoint = logdir + "/checkpoints"
    logfile = checkpoint + "/_metrics.json"

    # data
    num_samples, num_features = int(1e4), int(1e1)
    X = torch.rand(num_samples, num_features)
    y = torch.randint(0, 5, size=[num_samples])
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=32, num_workers=1)
    loaders = {
        "train": loader,
        "valid": loader,
    }

    # model, criterion, optimizer, scheduler
    model = torch.nn.Linear(num_features, 5)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    runner = SupervisedRunner()

    # first stage
    runner.train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        loaders=loaders,
        logdir=logdir,
        num_epochs=5,
        verbose=False,
        callbacks=[PeriodicLoaderCallback(valid=3)],
        load_best_on_end=True,
    )

    sys.stdout = old_stdout
    exp_output = str_stdout.getvalue()

    assert len(re.findall(r"\(train\)", exp_output)) == 5
    assert len(re.findall(r"\(valid\)", exp_output)) == 1
    assert (
        len(
            re.findall(r"\(global epoch 3, epoch 3, stage train\)", exp_output)
        )
        == 1
    )
    assert len(re.findall(r".*/train\.\d\.pth", exp_output)) == 1

    assert os.path.isfile(logfile)
    assert os.path.isfile(checkpoint + "/train.3.pth")
    assert os.path.isfile(checkpoint + "/best.pth")
    assert os.path.isfile(checkpoint + "/best_full.pth")
    assert os.path.isfile(checkpoint + "/last.pth")
    assert os.path.isfile(checkpoint + "/last_full.pth")

    shutil.rmtree(logdir, ignore_errors=True)
