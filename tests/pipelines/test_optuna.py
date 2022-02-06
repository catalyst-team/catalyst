# flake8: noqa

from tempfile import TemporaryDirectory

from pytest import mark

import torch
from torch import nn
from torch.utils.data import DataLoader

from catalyst import dl
from catalyst.contrib.datasets import MNIST
from catalyst.settings import IS_CUDA_AVAILABLE, NUM_CUDA_DEVICES, SETTINGS
from tests import (
    DATA_ROOT,
    IS_CPU_REQUIRED,
    IS_DDP_AMP_REQUIRED,
    IS_DDP_REQUIRED,
    IS_DP_AMP_REQUIRED,
    IS_DP_REQUIRED,
    IS_GPU_AMP_REQUIRED,
    IS_GPU_REQUIRED,
)

if SETTINGS.optuna_required:
    import optuna


def train_experiment(engine=None):
    with TemporaryDirectory() as logdir:

        def objective(trial):
            lr = trial.suggest_loguniform("lr", 1e-3, 1e-1)
            num_hidden = int(trial.suggest_loguniform("num_hidden", 32, 128))

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
                nn.Flatten(),
                nn.Linear(784, num_hidden),
                nn.ReLU(),
                nn.Linear(num_hidden, 10),
            )
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            criterion = nn.CrossEntropyLoss()

            runner = dl.SupervisedRunner(
                input_key="features", output_key="logits", target_key="targets"
            )
            runner.train(
                engine=engine,
                model=model,
                criterion=criterion,
                optimizer=optimizer,
                loaders=loaders,
                callbacks={
                    "optuna": dl.OptunaPruningCallback(
                        loader_key="valid",
                        metric_key="accuracy01",
                        minimize=False,
                        trial=trial,
                    ),
                    "accuracy": dl.AccuracyCallback(
                        input_key="logits", target_key="targets", num_classes=10
                    ),
                },
                num_epochs=2,
            )
            score = trial.best_score
            return score

        study = optuna.create_study(
            direction="maximize",
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=1, n_warmup_steps=0, interval_steps=1
            ),
        )
        study.optimize(objective, n_trials=3, timeout=300)
        print(study.best_value, study.best_params)


# Torch
@mark.skipif(not IS_CPU_REQUIRED, reason="CUDA device is not available")
def test_classification_on_cpu():
    train_experiment(dl.CPUEngine())


@mark.skipif(
    not all([IS_GPU_REQUIRED, IS_CUDA_AVAILABLE]), reason="CUDA device is not available"
)
def test_classification_on_torch_cuda0():
    train_experiment(dl.GPUEngine())


# @mark.skipif(
#     not (IS_CUDA_AVAILABLE and NUM_CUDA_DEVICES >= 2), reason="No CUDA>=2 found"
# )
# def test_classification_on_torch_cuda1():
#     train_experiment("cuda:1")


@mark.skipif(
    not all([IS_DP_REQUIRED, IS_CUDA_AVAILABLE, NUM_CUDA_DEVICES >= 2]),
    reason="No CUDA>=2 found",
)
def test_classification_on_torch_dp():
    train_experiment(dl.DataParallelEngine())


# @mark.skipif(
#     not all([IS_DDP_REQUIRED, IS_CUDA_AVAILABLE, NUM_CUDA_DEVICES >= 2]),
#     reason="No CUDA>=2 found",
# )
# def test_classification_on_torch_ddp():
#     train_experiment(dl.DistributedDataParallelEngine())


# AMP
@mark.skipif(
    not all([IS_GPU_AMP_REQUIRED, IS_CUDA_AVAILABLE, SETTINGS.amp_required]),
    reason="No CUDA or AMP found",
)
def test_classification_on_amp():
    train_experiment(dl.GPUEngine(fp16=True))


@mark.skipif(
    not all(
        [
            IS_DP_AMP_REQUIRED,
            IS_CUDA_AVAILABLE,
            NUM_CUDA_DEVICES >= 2,
            SETTINGS.amp_required,
        ]
    ),
    reason="No CUDA>=2 or AMP found",
)
def test_classification_on_amp_dp():
    train_experiment(dl.DataParallelEngine(fp16=True))


# @mark.skipif(
#     not all(
#         [
#             IS_DDP_AMP_REQUIRED,
#             IS_CUDA_AVAILABLE,
#             NUM_CUDA_DEVICES >= 2,
#             SETTINGS.amp_required,
#         ]
#     ),
#     reason="No CUDA>=2 or AMP found",
# )
# def test_classification_on_amp_ddp():
#     train_experiment(dl.DistributedDataParallelEngine(fp16=True))
