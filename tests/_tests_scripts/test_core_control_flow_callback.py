# flake8: noqa
from io import StringIO
import os
import re
import shutil
import sys

import pytest

import torch
from torch.utils.data import DataLoader, TensorDataset

import catalyst.dl as dl


def test_disabling_loss_for_validation_loader():
    old_stdout = sys.stdout
    sys.stdout = str_stdout = StringIO()

    # experiment_setup
    logdir = "./logs/control_flow"
    checkpoint = logdir + "/checkpoints"
    logfile = checkpoint + "/_metrics.json"

    # data
    num_samples, num_features = int(1e4), int(1e1)
    X = torch.rand(num_samples, num_features)
    y = torch.randint(0, 5, size=[num_samples])
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=32, num_workers=1)
    loaders = {"train": loader, "valid": loader}

    # model, criterion, optimizer, scheduler
    model = torch.nn.Linear(num_features, 5)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    runner = dl.SupervisedRunner()

    n_epochs = 5
    # first stage
    runner.train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        loaders=loaders,
        logdir=logdir,
        num_epochs=n_epochs,
        verbose=False,
        main_metric="accuracy01",
        callbacks=[
            dl.ControlFlowCallback(
                dl.CriterionCallback(), ignore_loaders=["valid"]
            ),
            dl.AccuracyCallback(accuracy_args=[1, 3, 5]),
            dl.CheckRunCallback(num_epoch_steps=n_epochs),
        ],
    )

    sys.stdout = old_stdout
    exp_output = str_stdout.getvalue()

    assert (
        len(
            re.findall(
                r"\(train\).* loss=\d+\.\d+$", exp_output, flags=re.MULTILINE
            )
        )
        == 5
    )
    assert (
        len(
            re.findall(
                r"\(valid\).* loss=\d+\.\d+$", exp_output, flags=re.MULTILINE
            )
        )
        == 0
    )
    assert (
        len(re.findall(r".*/train\.\d\.pth", exp_output, flags=re.MULTILINE))
        == 1
    )

    assert os.path.isfile(logfile)
    assert os.path.isfile(checkpoint + "/best.pth")
    assert os.path.isfile(checkpoint + "/best_full.pth")
    assert os.path.isfile(checkpoint + "/last.pth")
    assert os.path.isfile(checkpoint + "/last_full.pth")
    pth_files = [
        file for file in os.listdir(checkpoint) if file.endswith(".pth")
    ]
    assert len(pth_files) == 6

    shutil.rmtree(logdir, ignore_errors=True)


def test_disabling_metric_for_validation():
    old_stdout = sys.stdout
    sys.stdout = str_stdout = StringIO()

    # experiment_setup
    logdir = "./logs/control_flow"
    checkpoint = logdir + "/checkpoints"
    logfile = checkpoint + "/_metrics.json"

    # data
    num_samples, num_features = int(1e4), int(1e1)
    X = torch.rand(num_samples, num_features)
    y = torch.randint(0, 5, size=[num_samples])
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=32, num_workers=1)
    loaders = {"train": loader, "valid": loader}

    # model, criterion, optimizer, scheduler
    model = torch.nn.Linear(num_features, 5)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    runner = dl.SupervisedRunner()

    n_epochs = 5
    # first stage
    runner.train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        loaders=loaders,
        logdir=logdir,
        num_epochs=n_epochs,
        verbose=False,
        callbacks=[
            dl.ControlFlowCallback(
                dl.AccuracyCallback(accuracy_args=[1, 3, 5]),
                ignore_loaders="valid",
            ),
            dl.CheckRunCallback(num_epoch_steps=n_epochs),
        ],
    )

    sys.stdout = old_stdout
    exp_output = str_stdout.getvalue()

    assert (
        len(
            re.findall(
                r"\(train\).*loss=\d+\.\d+$", exp_output, flags=re.MULTILINE
            )
        )
        == 5
    )
    assert (
        len(
            re.findall(
                r"\(valid\): loss=\d+\.\d+$", exp_output, flags=re.MULTILINE
            )
        )
        == 5
    )
    assert (
        len(re.findall(r".*/train\.\d\.pth", exp_output, flags=re.MULTILINE))
        == 1
    )

    assert os.path.isfile(logfile)
    assert os.path.isfile(checkpoint + "/best.pth")
    assert os.path.isfile(checkpoint + "/best_full.pth")
    assert os.path.isfile(checkpoint + "/last.pth")
    assert os.path.isfile(checkpoint + "/last_full.pth")
    pth_files = [
        file for file in os.listdir(checkpoint) if file.endswith(".pth")
    ]
    assert len(pth_files) == 6

    shutil.rmtree(logdir, ignore_errors=True)


@pytest.mark.skip("loss should be specified for train loader")
def test_disabling_loss_for_train():
    old_stdout = sys.stdout
    sys.stdout = str_stdout = StringIO()

    # experiment_setup
    logdir = "./logs/control_flow"
    checkpoint = logdir + "/checkpoints"
    logfile = checkpoint + "/_metrics.json"

    # data
    num_samples, num_features = int(1e4), int(1e1)
    X = torch.rand(num_samples, num_features)
    y = torch.randint(0, 5, size=[num_samples])
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=32, num_workers=1)
    loaders = {"train": loader, "valid": loader}

    # model, criterion, optimizer, scheduler
    model = torch.nn.Linear(num_features, 5)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    runner = dl.SupervisedRunner()

    n_epochs = 5
    # first stage
    runner.train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        loaders=loaders,
        logdir=logdir,
        num_epochs=n_epochs,
        verbose=False,
        main_metric="accuracy01",
        callbacks=[
            dl.ControlFlowCallback(
                dl.CriterionCallback(), ignore_loaders=["train"]
            ),
            dl.AccuracyCallback(accuracy_args=[1, 3, 5]),
            dl.CheckRunCallback(num_epoch_steps=n_epochs),
        ],
    )

    sys.stdout = old_stdout
    exp_output = str_stdout.getvalue()

    assert len(re.findall(r"\(train\): loss", exp_output)) == 5
    assert len(re.findall(r"\(valid\): loss", exp_output)) == 0
    assert len(re.findall(r".*/train\.\d\.pth", exp_output)) == 1

    assert os.path.isfile(logfile)
    assert os.path.isfile(checkpoint + "/best.pth")
    assert os.path.isfile(checkpoint + "/best_full.pth")
    assert os.path.isfile(checkpoint + "/last.pth")
    assert os.path.isfile(checkpoint + "/last_full.pth")
    pth_files = [
        file for file in os.listdir(checkpoint) if file.endswith(".pth")
    ]
    assert len(pth_files) == 6

    shutil.rmtree(logdir, ignore_errors=True)


def test_ignoring_metric_on_train_dataset():
    old_stdout = sys.stdout
    sys.stdout = str_stdout = StringIO()

    # experiment_setup
    logdir = "./logs/control_flow"
    checkpoint = logdir + "/checkpoints"
    logfile = checkpoint + "/_metrics.json"

    # data
    num_samples, num_features = int(1e4), int(1e1)
    X = torch.rand(num_samples, num_features)
    y = torch.randint(0, 5, size=[num_samples])
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=32, num_workers=1)
    loaders = {"train": loader, "valid": loader}

    # model, criterion, optimizer, scheduler
    model = torch.nn.Linear(num_features, 5)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    runner = dl.SupervisedRunner()

    n_epochs = 5
    # first stage
    runner.train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        loaders=loaders,
        logdir=logdir,
        num_epochs=n_epochs,
        verbose=False,
        callbacks=[
            dl.ControlFlowCallback(
                dl.AccuracyCallback(accuracy_args=[1, 3, 5]),
                ignore_loaders="train",
            ),
            dl.CheckRunCallback(num_epoch_steps=n_epochs),
        ],
    )

    sys.stdout = old_stdout
    exp_output = str_stdout.getvalue()

    assert (
        len(
            re.findall(
                r"\(train\): loss=\d+\.\d+$", exp_output, flags=re.MULTILINE
            )
        )
        == 5
    )
    assert (
        len(
            re.findall(
                r"\(valid\): .*loss=\d+\.\d+$", exp_output, flags=re.MULTILINE
            )
        )
        == 5
    )
    assert (
        len(re.findall(r".*/train\.\d\.pth", exp_output, flags=re.MULTILINE))
        == 1
    )

    assert os.path.isfile(logfile)
    assert os.path.isfile(checkpoint + "/best.pth")
    assert os.path.isfile(checkpoint + "/best_full.pth")
    assert os.path.isfile(checkpoint + "/last.pth")
    assert os.path.isfile(checkpoint + "/last_full.pth")
    pth_files = [
        file for file in os.listdir(checkpoint) if file.endswith(".pth")
    ]
    assert len(pth_files) == 6

    shutil.rmtree(logdir, ignore_errors=True)
