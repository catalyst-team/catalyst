# flake8: noqa
# TODO: add test for `save_n_best=0``

import os
import re

import pytest

import torch
from torch.utils.data import DataLoader, TensorDataset

# local
import catalyst.dl as dl


def train_runner(logdir, n_epochs, callbacks):
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
    runner.loggers = {}

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
        callbacks=callbacks,
    )
    return runner


def test_files_existence(tmpdir):
    logfile = tmpdir + "/model.storage.json"
    n_epochs = 5
    callbacks = [
        dl.CheckpointCallback(
            logdir=tmpdir,
            loader_key="valid",
            metric_key="loss",
            minimize=True,
            topk=2,
        ),
        dl.CheckRunCallback(num_epoch_steps=n_epochs),
    ]
    train_runner(tmpdir, n_epochs, callbacks)

    assert os.path.isfile(logfile)
    assert os.path.isfile(tmpdir + "/model.0004.pth")
    # assert os.path.isfile(tmpdir + "/train.4_full.pth")
    assert os.path.isfile(tmpdir + "/model.0005.pth")
    # assert os.path.isfile(tmpdir + "/train.5_full.pth")
    assert os.path.isfile(tmpdir + "/model.best.pth")
    # assert os.path.isfile(tmpdir + "/best_full.pth")
    assert os.path.isfile(tmpdir + "/model.last.pth")
    # assert os.path.isfile(tmpdir + "/last_full.pth")


# @pytest.mark.parametrize(
#     ("to_load", "exp_loaded"), [("best", "model"), ("best_full", "full")]
# )
# def test_load_str_on_stage_end(to_load, exp_loaded, capsys, tmpdir):
#     # experiment_setup
#     n_epochs = 5
#     callbacks = [
#         dl.CheckpointCallback(
#             logdir=tmpdir,
#             loader_key="valid",
#             metric_key="loss",
#             minimize=True,
#             topk=2,
#             load_on_stage_end=to_load,
#         ),
#         dl.CheckRunCallback(num_epoch_steps=n_epochs),
#     ]

#     train_runner(tmpdir, n_epochs, callbacks)
#     exp_output = capsys.readouterr().out

#     assert len(re.findall(r"=> Loading", exp_output)) == 1
#     assert len(re.findall(r"=> Loading .*{}\.pth".format(to_load), exp_output)) == 1
#     assert len(re.findall(r"{} checkpoint".format(exp_loaded), exp_output)) == 1


# @pytest.mark.parametrize(
#     ("to_load", "exp_loaded"),
#     [
#         (
#             {"model": "best", "criterion": "best", "optimizer": "last"},
#             "model, criterion",
#         ),
#         (
#             {"model": "best", "criterion": "best", "optimizer": "best"},
#             "model, criterion, optimizer",
#         ),
#     ],
# )
# def test_load_dict_on_stage_end(to_load, exp_loaded, capsys, tmpdir):
#     # experiment_setup
#     n_epochs = 5
#     callbacks = [
#         dl.CheckpointCallback(
#             logdir=tmpdir,
#             loader_key="valid",
#             metric_key="loss",
#             minimize=True,
#             topk=2,
#             load_on_stage_end=to_load,
#         ),
#         dl.CheckRunCallback(num_epoch_steps=n_epochs),
#     ]

#     train_runner(tmpdir, n_epochs, callbacks)
#     exp_output = capsys.readouterr().out

#     assert len(re.findall(r"=> Loading", exp_output)) == 1
#     assert len(re.findall(r"loaded: {}".format(exp_loaded), exp_output)) == 1


# @pytest.mark.parametrize("to_load", [{}, None])
# def test_load_empty(to_load, capsys, tmpdir):
#     # experiment_setup
#     n_epochs = 5
#     callbacks = [
#         dl.CheckpointCallback(
#             logdir=tmpdir,
#             loader_key="valid",
#             metric_key="loss",
#             minimize=True,
#             topk=2,
#             load_on_stage_start=to_load,
#             load_on_stage_end=to_load,
#             resume=to_load,
#         ),
#         dl.CheckRunCallback(num_epoch_steps=n_epochs),
#     ]

#     train_runner(tmpdir, n_epochs, callbacks)
#     exp_output = capsys.readouterr().out

#     assert len(re.findall(r"=> Loading", exp_output)) == 0


# @pytest.mark.parametrize(
#     "to_load",
#     ["best", {"model": "not_existing_file.pth", "criterion": "not_existing_file.pth"}],
# )
# def test_resume_with_missing_file(to_load, tmpdir):
#     n_epochs = 5
#     callbacks = [
#         dl.CheckpointCallback(
#             logdir=tmpdir,
#             loader_key="valid",
#             metric_key="loss",
#             minimize=True,
#             topk=2,
#             load_on_stage_start=to_load,
#             load_on_stage_end=to_load,
#             resume="best",
#         ),
#         dl.CheckRunCallback(num_epoch_steps=n_epochs),
#     ]

#     with pytest.raises(FileNotFoundError):
#         train_runner(tmpdir, n_epochs, callbacks)
