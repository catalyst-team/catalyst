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


def test_load_best_on_stage_end():
    old_stdout = sys.stdout
    sys.stdout = str_stdout = StringIO()

    # experiment_setup
    logdir = "./logs/checkpoint_callback"
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
            dl.CheckpointCallback(save_n_best=2, load_on_stage_end="best"),
            dl.CheckRunCallback(num_epoch_steps=n_epochs),
        ],
    )

    sys.stdout = old_stdout
    exp_output = str_stdout.getvalue()

    assert len(re.findall(r"=> Loading", exp_output)) == 1
    assert len(re.findall(r"=> Loading .*best\.pth", exp_output)) == 1

    assert os.path.isfile(logfile)
    assert os.path.isfile(checkpoint + "/train.4.pth")
    assert os.path.isfile(checkpoint + "/train.4_full.pth")
    assert os.path.isfile(checkpoint + "/train.5.pth")
    assert os.path.isfile(checkpoint + "/train.5_full.pth")
    assert os.path.isfile(checkpoint + "/best.pth")
    assert os.path.isfile(checkpoint + "/best_full.pth")
    assert os.path.isfile(checkpoint + "/last.pth")
    assert os.path.isfile(checkpoint + "/last_full.pth")

    shutil.rmtree(logdir, ignore_errors=True)


def test_multiple_stages_and_different_checkpoints_to_load():
    old_stdout = sys.stdout
    sys.stdout = str_stdout = StringIO()

    # experiment_setup
    logdir = "./logs/checkpoint_callback"
    checkpoint = logdir + "/checkpoints"
    logfile = checkpoint + "/_metrics.json"
    num_epochs = 5

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

    # first stage
    runner.train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        loaders=loaders,
        logdir=logdir,
        num_epochs=num_epochs,
        verbose=False,
        callbacks=[
            dl.CheckpointCallback(
                save_n_best=2,
                load_on_stage_end={
                    "model": "best",
                    "criterion": "best",
                    "optimizer": "last",
                },
            ),
            dl.CheckRunCallback(num_epoch_steps=num_epochs),
        ],
    )
    # second stage
    runner.train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        loaders=loaders,
        logdir=logdir,
        num_epochs=num_epochs,
        verbose=False,
        callbacks=[
            dl.CheckpointCallback(
                save_n_best=3,
                load_on_stage_start={
                    "model": "last",
                    "criterion": "last",
                    "optimizer": "best",
                },
            ),
            dl.CheckRunCallback(num_epoch_steps=num_epochs),
        ],
    )

    sys.stdout = old_stdout
    exp_output = str_stdout.getvalue()

    assert len(re.findall(r"=> Loading", exp_output)) == 3
    assert len(re.findall(r"=> Loading .*best_full\.pth", exp_output)) == 2
    assert len(re.findall(r"=> Loading .*last_full\.pth", exp_output)) == 1

    assert os.path.isfile(logfile)
    assert os.path.isfile(checkpoint + "/train.3.pth")
    assert os.path.isfile(checkpoint + "/train.3_full.pth")
    assert os.path.isfile(checkpoint + "/train.4.pth")
    assert os.path.isfile(checkpoint + "/train.4_full.pth")
    assert os.path.isfile(checkpoint + "/train.5.pth")
    assert os.path.isfile(checkpoint + "/train.5_full.pth")
    assert os.path.isfile(checkpoint + "/best.pth")
    assert os.path.isfile(checkpoint + "/best_full.pth")
    assert os.path.isfile(checkpoint + "/last.pth")
    assert os.path.isfile(checkpoint + "/last_full.pth")

    shutil.rmtree(logdir, ignore_errors=True)


def test_resume_with_missing_file():
    old_stdout = sys.stdout
    sys.stdout = str_stdout = StringIO()

    # experiment_setup
    logdir = "./logs/checkpoint_callback"
    checkpoint = logdir + "/checkpoints"
    logfile = checkpoint + "/_metrics.json"
    num_epochs = 5

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

    with pytest.raises(FileNotFoundError):
        runner.train(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            loaders=loaders,
            logdir=logdir,
            num_epochs=num_epochs,
            verbose=False,
            callbacks=[
                dl.CheckpointCallback(
                    save_n_best=2,
                    load_on_stage_end={
                        "model": "best",
                        "criterion": "best",
                        "optimizer": "last",
                    },
                    resume="not_existing_file.pth",
                ),
                dl.CheckRunCallback(num_epoch_steps=num_epochs),
            ],
        )

    sys.stdout = old_stdout
    exp_output = str_stdout.getvalue()

    shutil.rmtree(logdir, ignore_errors=True)


def test_load_on_stage_start_with_empty_dict():
    old_stdout = sys.stdout
    sys.stdout = str_stdout = StringIO()

    # experiment_setup
    logdir = "./logs/checkpoint_callback"
    checkpoint = logdir + "/checkpoints"
    logfile = checkpoint + "/_metrics.json"
    num_epochs = 5

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

    # first stage
    runner.train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        loaders=loaders,
        logdir=logdir,
        num_epochs=num_epochs,
        verbose=False,
        callbacks=[
            dl.CheckpointCallback(save_n_best=2),
            dl.CheckRunCallback(num_epoch_steps=num_epochs),
        ],
    )
    # second stage
    runner.train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        loaders=loaders,
        logdir=logdir,
        num_epochs=num_epochs,
        verbose=False,
        callbacks=[
            dl.CheckpointCallback(save_n_best=3, load_on_stage_start={}),
            dl.CheckRunCallback(num_epoch_steps=num_epochs),
        ],
    )

    sys.stdout = old_stdout
    exp_output = str_stdout.getvalue()

    assert len(re.findall(r"=> Loading", exp_output)) == 0

    assert os.path.isfile(logfile)
    assert os.path.isfile(checkpoint + "/train.3.pth")
    assert os.path.isfile(checkpoint + "/train.3_full.pth")
    assert os.path.isfile(checkpoint + "/train.4.pth")
    assert os.path.isfile(checkpoint + "/train.4_full.pth")
    assert os.path.isfile(checkpoint + "/train.5.pth")
    assert os.path.isfile(checkpoint + "/train.5_full.pth")
    assert os.path.isfile(checkpoint + "/best.pth")
    assert os.path.isfile(checkpoint + "/best_full.pth")
    assert os.path.isfile(checkpoint + "/last.pth")
    assert os.path.isfile(checkpoint + "/last_full.pth")

    shutil.rmtree(logdir, ignore_errors=True)
