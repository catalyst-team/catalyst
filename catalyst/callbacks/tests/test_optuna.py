# flake8: noqa

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from catalyst import dl
from catalyst.contrib.datasets import MNIST
from catalyst.contrib.nn.modules import Flatten
from catalyst.data.transforms import ToTensor
from catalyst.dl import AccuracyCallback
from catalyst.settings import SETTINGS

if SETTINGS.optuna_required:
    import optuna

    from catalyst.callbacks import OptunaPruningCallback


@pytest.mark.skipif(
    not (SETTINGS.optuna_required), reason="No optuna required",
)
def test_optuna():
    trainset = MNIST("./data", train=False, download=True, transform=ToTensor())
    testset = MNIST("./data", train=False, download=True, transform=ToTensor())
    loaders = {
        "train": DataLoader(trainset, batch_size=32),
        "valid": DataLoader(testset, batch_size=64),
    }
    model = nn.Sequential(Flatten(), nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 10))

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
            callbacks={
                "optuna": OptunaPruningCallback(
                    loader_key="valid", metric_key="loss", minimize=True, trial=trial
                ),
                "accuracy": AccuracyCallback(
                    num_classes=10, input_key="logits", target_key="targets"
                ),
            },
            num_epochs=2,
            valid_metric="accuracy01",
            minimize_valid_metric=False,
        )
        return runner.callbacks["optuna"].best_score

    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=1, n_warmup_steps=0, interval_steps=1),
    )
    study.optimize(objective, n_trials=2, timeout=300)
    assert True
