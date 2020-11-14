# flake8: noqa
import optuna

import torch
from torch import nn
from torch.utils.data import DataLoader

from catalyst import dl
from catalyst.contrib.datasets import MNIST
from catalyst.contrib.dl.callbacks import OptunaCallback
from catalyst.contrib.nn import Flatten
from catalyst.data.cv.transforms.torch import ToTensor
from catalyst.dl import AccuracyCallback


def test_mnist():
    trainset = MNIST(
        "./data", train=False, download=True, transform=ToTensor(),
    )
    testset = MNIST(
        "./data", train=False, download=True, transform=ToTensor(),
    )
    loaders = {
        "train": DataLoader(trainset, batch_size=32),
        "valid": DataLoader(testset, batch_size=64),
    }
    model = nn.Sequential(
        Flatten(), nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 10)
    )

    def objective(trial):
        lr = trial.suggest_loguniform("lr", 1e-3, 1e-1)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        runner = dl.SupervisedRunner()
        runner.train(
            model=model,
            loaders=loaders,
            criterion=criterion,
            optimizer=optimizer,
            callbacks=[
                OptunaCallback(trial),
                AccuracyCallback(num_classes=10),
            ],
            num_epochs=10,
            main_metric="accuracy01",
            minimize_metric=False,
        )
        return runner.best_valid_metrics[runner.main_metric]

    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=1, n_warmup_steps=0, interval_steps=1
        ),
    )
    study.optimize(objective, n_trials=5, timeout=300)
    assert True
