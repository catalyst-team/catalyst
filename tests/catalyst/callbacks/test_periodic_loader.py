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
    CheckpointCallback,
    CheckRunCallback,
    CriterionCallback,
    PeriodicLoaderCallback,
    SupervisedRunner,
)


@pytest.mark.skip(reason="disabled")
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
                msg = f"Epochs (epoch={runner.epoch}) " "without valid loader can't be best!"
                assert not runner.is_best_valid, msg
            else:
                assert runner.valid_metrics[runner.valid_metric] is not None
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
        valid_loader="valid",
        valid_metric="loss",
        minimize_valid_metric=True,
        callbacks=[
            PeriodicLoaderCallback(
                valid_loader_key="valid", valid_metric_key="loss", minimize=True, valid=2
            ),
            BestStateCheckerCallback(),
            CheckRunCallback(num_epoch_steps=5),
        ],
    )

    # # second stage
    # runner.train(
    #     model=model,
    #     criterion=criterion,
    #     optimizer=optimizer,
    #     loaders=loaders,
    #     logdir=logdir,
    #     num_epochs=6,
    #     verbose=False,
    #     callbacks=[
    #         PeriodicLoaderCallback(valid=3),
    #         BestStateCheckerCallback(),
    #         CheckRunCallback(num_epoch_steps=6),
    #     ],
    # )
    #
    # # third stage
    # runner.train(
    #     model=model,
    #     criterion=criterion,
    #     optimizer=optimizer,
    #     loaders=loaders,
    #     logdir=logdir,
    #     num_epochs=6,
    #     verbose=False,
    #     callbacks=[
    #         PeriodicLoaderCallback(valid=4),
    #         BestStateCheckerCallback(),
    #         CheckRunCallback(num_epoch_steps=6),
    #     ],
    # )

    shutil.rmtree(logdir, ignore_errors=True)


@pytest.mark.skip(reason="disabled")
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
        valid_loader="valid",
        valid_metric="loss",
        minimize_valid_metric=True,
        callbacks=[
            PeriodicLoaderCallback(
                valid_loader_key="valid", valid_metric_key="loss", minimize=True, valid=3
            ),
            CheckRunCallback(num_epoch_steps=10),
        ],
    )

    sys.stdout = old_stdout
    exp_output = str_stdout.getvalue()

    # assert len(re.findall(r"\(train\)", exp_output)) == 10
    # assert len(re.findall(r"\(valid\)", exp_output)) == 3
    # assert len(re.findall(r".*/train\.\d\.pth", exp_output)) == 1

    assert os.path.isfile(logfile)
    assert os.path.isfile(checkpoint + "/train.9.pth")
    assert os.path.isfile(checkpoint + "/train.9_full.pth")
    assert os.path.isfile(checkpoint + "/best.pth")
    assert os.path.isfile(checkpoint + "/best_full.pth")
    assert os.path.isfile(checkpoint + "/last.pth")
    assert os.path.isfile(checkpoint + "/last_full.pth")

    shutil.rmtree(logdir, ignore_errors=True)


@pytest.mark.skip(reason="disabled support period = 0 for validation loaders")
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
        valid_loader="valid",
        valid_metric="loss",
        minimize_valid_metric=True,
        callbacks=[
            PeriodicLoaderCallback(
                valid_loader_key="valid", valid_metric_key="loss", minimize=True, valid=0
            ),
            CheckRunCallback(num_epoch_steps=5),
        ],
    )

    sys.stdout = old_stdout
    exp_output = str_stdout.getvalue()

    # assert len(re.findall(r"\(train\)", exp_output)) == 5
    # assert len(re.findall(r"\(valid\)", exp_output)) == 0
    # assert len(re.findall(r".*/train\.\d\.pth", exp_output)) == 1

    assert os.path.isfile(logfile)
    assert os.path.isfile(checkpoint + "/train.5.pth")
    assert os.path.isfile(checkpoint + "/train.5_full.pth")
    assert os.path.isfile(checkpoint + "/best.pth")
    assert os.path.isfile(checkpoint + "/best_full.pth")
    assert os.path.isfile(checkpoint + "/last.pth")
    assert os.path.isfile(checkpoint + "/last_full.pth")

    shutil.rmtree(logdir, ignore_errors=True)


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
        valid_loader="valid",
        valid_metric="loss",
        minimize_valid_metric=True,
        callbacks=[
            PeriodicLoaderCallback(
                valid_loader_key="valid",
                valid_metric_key="loss",
                minimize=True,
                train_additional=2,
                valid=3,
                valid_additional=0,
            ),
            CheckRunCallback(num_epoch_steps=10),
        ],
    )

    sys.stdout = old_stdout
    exp_output = str_stdout.getvalue()

    # assert len(re.findall(r"\(train\)", exp_output)) == 10
    # assert len(re.findall(r"\(train_additional\)", exp_output)) == 5
    # assert len(re.findall(r"\(valid\)", exp_output)) == 3
    # assert len(re.findall(r"\(valid_additional\)", exp_output)) == 0
    # assert len(re.findall(r".*/train\.\d\.pth", exp_output)) == 1

    assert os.path.isfile(logfile)
    assert os.path.isfile(checkpoint + "/train.9.pth")
    assert os.path.isfile(checkpoint + "/train.9_full.pth")
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
        valid_loader="valid",
        valid_metric="loss",
        minimize_valid_metric=True,
        callbacks=[
            PeriodicLoaderCallback(
                valid_loader_key="valid",
                valid_metric_key="loss",
                minimize=True,
                train_additional=2,
                valid=3,
                valid_additional=0,
            ),
            CheckRunCallback(num_epoch_steps=5),
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
        valid_loader="valid",
        valid_metric="loss",
        minimize_valid_metric=True,
        callbacks=[
            PeriodicLoaderCallback(
                valid_loader_key="valid",
                valid_metric_key="loss",
                minimize=True,
                train_additional=2,
                valid=3,
                valid_additional=0,
            ),
            CheckRunCallback(num_epoch_steps=10),
        ],
    )

    sys.stdout = old_stdout
    exp_output = str_stdout.getvalue()

    # assert len(re.findall(r"\(train\)", exp_output)) == 15
    # assert len(re.findall(r"\(train_additional\)", exp_output)) == 7
    # assert len(re.findall(r"\(valid\)", exp_output)) == 4
    # assert len(re.findall(r"\(valid_additional\)", exp_output)) == 0
    # assert len(re.findall(r".*/train\.\d\.pth", exp_output)) == 2

    assert os.path.isfile(logfile)
    assert os.path.isfile(checkpoint + "/train.9.pth")
    assert os.path.isfile(checkpoint + "/train.9_full.pth")
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
            valid_loader="valid",
            valid_metric="loss",
            minimize_valid_metric=True,
            callbacks=[
                PeriodicLoaderCallback(
                    valid_loader_key="valid",
                    valid_metric_key="loss",
                    minimize=True,
                    train=2,
                    train_additional=2,
                    valid=3,
                    valid_additional=0,
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
            valid_loader="valid",
            valid_metric="loss",
            minimize_valid_metric=True,
            callbacks=[
                PeriodicLoaderCallback(
                    valid_loader_key="valid",
                    valid_metric_key="loss",
                    minimize=True,
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


def test_negative_period_exception():
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
            valid_loader="valid",
            valid_metric="loss",
            minimize_valid_metric=True,
            callbacks=[
                PeriodicLoaderCallback(
                    valid_loader_key="valid",
                    valid_metric_key="loss",
                    minimize=True,
                    train_additional=1,
                    train_not_exists=-10,
                    valid=3,
                    valid_additional=-1,
                    valid_not_exist=1,
                )
            ],
        )

    sys.stdout = old_stdout
    exp_output = str_stdout.getvalue()

    shutil.rmtree(logdir, ignore_errors=True)


def test_zero_period_validation_exception():
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
            valid_loader="valid",
            valid_metric="loss",
            minimize_valid_metric=True,
            callbacks=[
                PeriodicLoaderCallback(
                    valid_loader_key="valid",
                    valid_metric_key="loss",
                    minimize=True,
                    train_additional=1,
                    train_not_exists=3,
                    valid=0,
                    valid_additional=2,
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
        valid_loader="valid",
        valid_metric="loss",
        minimize_valid_metric=True,
        callbacks=[
            PeriodicLoaderCallback(
                valid_loader_key="valid",
                valid_metric_key="loss",
                minimize=True,
                train_additional=2,
                train_not_exists=2,
                valid=3,
                valid_additional=0,
                valid_not_exist=1,
            ),
            CheckRunCallback(num_epoch_steps=10),
        ],
    )

    sys.stdout = old_stdout
    exp_output = str_stdout.getvalue()

    # assert len(re.findall(r"\(train\)", exp_output)) == 10
    # assert len(re.findall(r"\(train_additional\)", exp_output)) == 5
    # assert len(re.findall(r"\(train_not_exists\)", exp_output)) == 0
    # assert len(re.findall(r"\(valid\)", exp_output)) == 3
    # assert len(re.findall(r"\(valid_additional\)", exp_output)) == 0
    # assert len(re.findall(r"\(valid_not_exist\)", exp_output)) == 0
    # assert len(re.findall(r".*/train\.\d\.pth", exp_output)) == 1

    assert os.path.isfile(logfile)
    assert os.path.isfile(checkpoint + "/train.9.pth")
    assert os.path.isfile(checkpoint + "/train.9_full.pth")
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
        valid_loader="valid",
        valid_metric="loss",
        minimize_valid_metric=True,
        callbacks=[
            PeriodicLoaderCallback(
                valid_loader_key="valid", valid_metric_key="loss", minimize=True, valid=3
            ),
            CheckRunCallback(num_epoch_steps=5),
        ],
        load_best_on_end=True,
    )

    sys.stdout = old_stdout
    exp_output = str_stdout.getvalue()

    # assert len(re.findall(r"\(train\)", exp_output)) == 5
    # assert len(re.findall(r"\(valid\)", exp_output)) == 1
    # assert len(re.findall(r"\(global epoch 3, epoch 3, stage train\)", exp_output)) == 1
    # assert len(re.findall(r".*/train\.\d\.pth", exp_output)) == 1

    assert os.path.isfile(logfile)
    assert os.path.isfile(checkpoint + "/train.3.pth")
    assert os.path.isfile(checkpoint + "/train.3_full.pth")
    assert os.path.isfile(checkpoint + "/best.pth")
    assert os.path.isfile(checkpoint + "/best_full.pth")
    assert os.path.isfile(checkpoint + "/last.pth")
    assert os.path.isfile(checkpoint + "/last_full.pth")

    shutil.rmtree(logdir, ignore_errors=True)


@pytest.mark.skip(reason="disabled")
def test_loading_best_state_at_end_with_custom_scores():
    class Metric(Callback):
        def __init__(self, values):
            super().__init__(CallbackOrder.metric)
            self.values = values

        def on_loader_end(self, runner: "IRunner") -> None:
            score = self.values[runner.loader_key][runner.stage_epoch_step]
            runner.loader_metrics["metric"] = score

    old_stdout = sys.stdout
    sys.stdout = str_stdout = StringIO()

    # experiment_setup
    logdir = "./logs/periodic_loader"
    checkpoint = logdir  # + "/checkpoints"
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

    n_epochs = 10
    period = 3
    metrics = {
        "train": {i: i * 0.1 for i in range(1, 11)},
        "valid": {
            i: v
            for i, v in enumerate([0.05, 0.1, 0.15, 0.15, 0.2, 0.18, 0.22, 0.11, 0.13, 0.12], 1)
        },
    }

    # first stage
    runner.train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        loaders=loaders,
        logdir=logdir,
        num_epochs=n_epochs,
        verbose=False,
        valid_loader="valid",
        valid_metric="metric",
        minimize_valid_metric=False,
        callbacks=[
            PeriodicLoaderCallback(
                valid_loader_key="valid", valid_metric_key="metric", minimize=True, valid=period
            ),
            CheckRunCallback(num_epoch_steps=n_epochs),
            Metric(metrics),
        ],
        load_best_on_end=True,
    )

    sys.stdout = old_stdout
    exp_output = str_stdout.getvalue()

    # assert len(re.findall(r"\(train\)", exp_output)) == n_epochs
    # assert len(re.findall(r"\(valid\)", exp_output)) == (n_epochs // period)
    # assert len(re.findall(r"\(global epoch 6, epoch 6, stage train\)", exp_output)) == 1
    # assert len(re.findall(r".*/train\.\d\.pth", exp_output)) == 1

    assert os.path.isfile(logfile)
    assert os.path.isfile(checkpoint + "/train.6.pth")
    assert os.path.isfile(checkpoint + "/train.6_full.pth")
    assert os.path.isfile(checkpoint + "/best.pth")
    assert os.path.isfile(checkpoint + "/best_full.pth")
    assert os.path.isfile(checkpoint + "/last.pth")
    assert os.path.isfile(checkpoint + "/last_full.pth")

    shutil.rmtree(logdir, ignore_errors=True)


@pytest.mark.skip(reason="disabled")
def test_multiple_best_checkpoints():
    old_stdout = sys.stdout
    sys.stdout = str_stdout = StringIO()

    # experiment_setup
    logdir = "./logs/periodic_loader"
    checkpoint = logdir  # + "/checkpoints"
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

    n_epochs = 12
    period = 2
    # first stage
    runner.train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        loaders=loaders,
        logdir=logdir,
        num_epochs=n_epochs,
        verbose=False,
        valid_loader="valid",
        valid_metric="loss",
        minimize_valid_metric=True,
        callbacks=[
            PeriodicLoaderCallback(
                valid_loader_key="valid", valid_metric_key="loss", minimize=True, valid=period
            ),
            CheckRunCallback(num_epoch_steps=n_epochs),
            CheckpointCallback(
                logdir=logdir, loader_key="valid", metric_key="loss", minimize=True, save_n_best=3
            ),
        ],
    )

    sys.stdout = old_stdout
    exp_output = str_stdout.getvalue()

    # assert len(re.findall(r"\(train\)", exp_output)) == n_epochs
    # assert len(re.findall(r"\(valid\)", exp_output)) == (n_epochs // period)
    # assert len(re.findall(r".*/train\.\d{1,2}\.pth", exp_output)) == 3

    assert os.path.isfile(logfile)
    assert os.path.isfile(checkpoint + "/train.8.pth")
    assert os.path.isfile(checkpoint + "/train.8_full.pth")
    assert os.path.isfile(checkpoint + "/train.10.pth")
    assert os.path.isfile(checkpoint + "/train.10_full.pth")
    assert os.path.isfile(checkpoint + "/train.12.pth")
    assert os.path.isfile(checkpoint + "/train.12_full.pth")
    assert os.path.isfile(checkpoint + "/best.pth")
    assert os.path.isfile(checkpoint + "/best_full.pth")
    assert os.path.isfile(checkpoint + "/last.pth")
    assert os.path.isfile(checkpoint + "/last_full.pth")

    shutil.rmtree(logdir, ignore_errors=True)
