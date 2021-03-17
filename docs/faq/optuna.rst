Optuna integration
==============================================================================

Notebook API
----------------------------------------------------

You can easily use Optuna for hyperparameters optimization:

.. code-block:: python

    import os
    import optuna
    import torch
    from torch import nn
    from torch.utils.data import DataLoader
    from catalyst import dl
    from catalyst.data.cv import ToTensor
    from catalyst.contrib.datasets import MNIST
    from catalyst.contrib.nn import Flatten


    def objective(trial):
        lr = trial.suggest_loguniform("lr", 1e-3, 1e-1)
        num_hidden = int(trial.suggest_loguniform("num_hidden", 32, 128))

        loaders = {
            "train": DataLoader(MNIST(os.getcwd(), train=True, download=True, transform=ToTensor()), batch_size=32),
            "valid": DataLoader(MNIST(os.getcwd(), train=False, download=True, transform=ToTensor()), batch_size=32),
        }
        model = nn.Sequential(
            Flatten(), nn.Linear(784, num_hidden), nn.ReLU(), nn.Linear(num_hidden, 10)
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        runner = dl.SupervisedRunner()
        runner.train(
            model=model,
            loaders=loaders,
            criterion=criterion,
            optimizer=optimizer,
            callbacks=[
                dl.OptunaCallback(trial),
                dl.AccuracyCallback(num_classes=10),
            ],
            num_epochs=10,
            valid_metric="accuracy01",
            minimize_metric=False,
        )
        return runner.best_valid_metrics[runner.valid_metric]

    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=1, n_warmup_steps=0, interval_steps=1
        ),
    )
    study.optimize(objective, n_trials=10, timeout=300)
    print(study.best_value, study.best_params)

Config API
----------------------------------------------------

Firstly, prepare the Optuna-based config. For example, like:

.. code-block:: yaml

    model_params:
        model: SimpleNet
        num_filters1: "int(trial.suggest_loguniform('num_filters1', 4, 32))"
        num_filters2: "int(trial.suggest_loguniform('num_filters2', 4, 32))"
        num_hiddens1: "int(trial.suggest_loguniform('num_hiddens1', 32, 128))"
        num_hiddens2: "int(trial.suggest_loguniform('num_hiddens2', 32, 128))"
        ...

After that just run:

.. code-block:: bash

    catalyst-dl tune --config=/path/to/config.yml --verbose

You also can visualize current training progress with:

.. code-block:: bash

    CUDA_VISIBLE_DEVICE="" tensorboard --logdir=/path/to/logdir


If you haven't found the answer for your question, feel free to `join our slack`_ for the discussion.

.. _`join our slack`: https://join.slack.com/t/catalyst-team-core/shared_invite/zt-d9miirnn-z86oKDzFMKlMG4fgFdZafw
