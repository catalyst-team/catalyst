# flake8: noqa
import torch
from torch.utils.data import DataLoader, TensorDataset

from catalyst.contrib.nn.schedulers.onecycle import OneCycleLRWithWarmup
from catalyst.dl import Callback, CallbackOrder, SupervisedRunner


class LRCheckerCallback(Callback):
    def __init__(self, init_lr_value: float, final_lr_value: float):
        super().__init__(CallbackOrder.Internal)
        self.init_lr = init_lr_value
        self.final_lr = final_lr_value

    # Check initial LR
    def on_batch_start(self, runner):
        step = getattr(runner, "global_batch_step")
        if step == 1:
            assert self.init_lr == runner.scheduler.get_lr()[0]

    # Check final LR
    def on_stage_end(self, runner):
        assert self.final_lr == runner.scheduler.get_lr()[0]


def test_onecyle():
    # experiment_setup
    logdir = "./logs/core_runner"

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

    # number of steps, epochs, LR range, initial LR and warmup_fraction
    num_steps = 6
    epochs = 8
    min_lr = 1e-4
    max_lr = 2e-3
    init_lr = 1e-3
    warmup_fraction = 0.5

    # model, criterion, optimizer, scheduler
    model = torch.nn.Linear(num_features, 5)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    scheduler = OneCycleLRWithWarmup(
        optimizer,
        num_steps=num_steps,
        lr_range=(max_lr, min_lr),
        init_lr=init_lr,
        warmup_fraction=warmup_fraction,
    )

    runner = SupervisedRunner()

    callbacks = [LRCheckerCallback(init_lr, min_lr)]

    # Single stage
    runner.train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        loaders=loaders,
        logdir=logdir,
        num_epochs=epochs,
        verbose=False,
        callbacks=callbacks,
    )
