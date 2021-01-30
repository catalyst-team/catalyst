from collections import OrderedDict

import torch

from catalyst.callbacks import (
    CheckpointCallback,
    CheckRunCallback,
    ConsoleLogger,
    CriterionCallback,
    ExceptionCallback,
    MetricManagerCallback,
    OptimizerCallback,
    SchedulerCallback,
    TensorboardLogger,
    TimerCallback,
    ValidationManagerCallback,
    VerboseLogger,
)
from catalyst.experiments.auto import AutoCallbackExperiment


def _test_callbacks(test_callbacks, exp, stage="train"):
    exp_callbacks = exp.get_callbacks(stage)
    exp_callbacks = OrderedDict(sorted(exp_callbacks.items(), key=lambda t: t[0]))
    test_callbacks = OrderedDict(sorted(test_callbacks.items(), key=lambda t: t[0]))

    assert exp_callbacks.keys() == test_callbacks.keys()
    cbs = zip(exp_callbacks.values(), test_callbacks.values())
    for callback, klass in cbs:
        assert isinstance(callback, klass)


def test_defaults():
    """
    Test on defaults for SupervisedExperiment class, which is child class of
    BaseExperiment.  That's why we check only default callbacks functionality
    here
    """
    model = torch.nn.Module()
    dataset = torch.utils.data.Dataset()
    dataloader = torch.utils.data.DataLoader(dataset)
    loaders = OrderedDict()
    loaders["train"] = dataloader

    test_callbacks = OrderedDict(
        [
            ("_metrics", MetricManagerCallback),
            ("_validation", ValidationManagerCallback),
            ("_console", ConsoleLogger),
            ("_exception", ExceptionCallback),
        ]
    )

    exp = AutoCallbackExperiment(model=model, loaders=loaders, valid_loader="train",)
    _test_callbacks(test_callbacks, exp)


def test_defaults_verbose():
    """Docs? Contribution is welcome."""
    test_callbacks = OrderedDict(
        [
            ("_verbose", VerboseLogger),
            ("_metrics", MetricManagerCallback),
            ("_validation", ValidationManagerCallback),
            ("_saver", CheckpointCallback),
            ("_console", ConsoleLogger),
            ("_tensorboard", TensorboardLogger),
            ("_exception", ExceptionCallback),
        ]
    )

    model = torch.nn.Module()
    dataset = torch.utils.data.Dataset()
    dataloader = torch.utils.data.DataLoader(dataset)
    loaders = OrderedDict()
    loaders["train"] = dataloader

    exp = AutoCallbackExperiment(
        model=model, loaders=loaders, verbose=True, valid_loader="train", logdir="./logs",
    )
    _test_callbacks(test_callbacks, exp)


def test_defaults_check():
    """Docs? Contribution is welcome."""
    test_callbacks = OrderedDict(
        [
            ("_check", CheckRunCallback),
            ("_metrics", MetricManagerCallback),
            ("_validation", ValidationManagerCallback),
            ("_saver", CheckpointCallback),
            ("_console", ConsoleLogger),
            ("_tensorboard", TensorboardLogger),
            ("_exception", ExceptionCallback),
        ]
    )

    model = torch.nn.Module()
    dataset = torch.utils.data.Dataset()
    dataloader = torch.utils.data.DataLoader(dataset)
    loaders = OrderedDict()
    loaders["train"] = dataloader

    exp = AutoCallbackExperiment(
        model=model, loaders=loaders, check_run=True, valid_loader="train", logdir="./logs",
    )
    _test_callbacks(test_callbacks, exp)


def test_criterion():
    """Docs? Contribution is welcome."""
    test_callbacks = OrderedDict(
        [
            ("_metrics", MetricManagerCallback),
            ("_validation", ValidationManagerCallback),
            ("_saver", CheckpointCallback),
            ("_console", ConsoleLogger),
            ("_tensorboard", TensorboardLogger),
            ("_exception", ExceptionCallback),
            ("_criterion", CriterionCallback),
        ]
    )

    model = torch.nn.Linear(10, 10)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = None
    scheduler = None
    dataset = torch.utils.data.Dataset()
    dataloader = torch.utils.data.DataLoader(dataset)
    loaders = OrderedDict()
    loaders["train"] = dataloader

    exp = AutoCallbackExperiment(
        model=model,
        loaders=loaders,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        valid_loader="train",
        logdir="./logs",
    )
    _test_callbacks(test_callbacks, exp)


def test_optimizer():
    """Docs? Contribution is welcome."""
    test_callbacks = OrderedDict(
        [
            ("_metrics", MetricManagerCallback),
            ("_validation", ValidationManagerCallback),
            ("_saver", CheckpointCallback),
            ("_console", ConsoleLogger),
            ("_tensorboard", TensorboardLogger),
            ("_exception", ExceptionCallback),
            ("_optimizer", OptimizerCallback),
        ]
    )

    model = torch.nn.Linear(10, 10)
    criterion = None
    optimizer = torch.optim.Adam(model.parameters())
    scheduler = None
    dataset = torch.utils.data.Dataset()
    dataloader = torch.utils.data.DataLoader(dataset)
    loaders = OrderedDict()
    loaders["train"] = dataloader

    exp = AutoCallbackExperiment(
        model=model,
        loaders=loaders,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        valid_loader="train",
        logdir="./logs",
    )
    _test_callbacks(test_callbacks, exp)


def test_scheduler():
    """Docs? Contribution is welcome."""
    test_callbacks = OrderedDict(
        [
            ("_metrics", MetricManagerCallback),
            ("_validation", ValidationManagerCallback),
            ("_saver", CheckpointCallback),
            ("_timer", TimerCallback),
            ("_console", ConsoleLogger),
            ("_tensorboard", TensorboardLogger),
            ("_exception", ExceptionCallback),
            ("_optimizer", OptimizerCallback),
            ("_scheduler", SchedulerCallback),
        ]
    )

    model = torch.nn.Linear(10, 10)
    optimizer = torch.optim.Adam(model.parameters())
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10)
    dataset = torch.utils.data.Dataset()
    dataloader = torch.utils.data.DataLoader(dataset)
    loaders = OrderedDict()
    loaders["train"] = dataloader

    exp = AutoCallbackExperiment(
        model=model,
        loaders=loaders,
        optimizer=optimizer,
        scheduler=scheduler,
        valid_loader="train",
        logdir="./logs",
        check_time=True,
    )
    _test_callbacks(test_callbacks, exp)


def test_all():
    """Docs? Contribution is welcome."""
    test_callbacks = OrderedDict(
        [
            ("_verbose", VerboseLogger),
            ("_check", CheckRunCallback),
            ("_metrics", MetricManagerCallback),
            ("_validation", ValidationManagerCallback),
            ("_console", ConsoleLogger),
            ("_exception", ExceptionCallback),
            ("_criterion", CriterionCallback),
            ("_optimizer", OptimizerCallback),
            ("_scheduler", SchedulerCallback),
        ]
    )

    model = torch.nn.Linear(10, 10)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10)
    dataset = torch.utils.data.Dataset()
    dataloader = torch.utils.data.DataLoader(dataset)
    loaders = OrderedDict()
    loaders["train"] = dataloader

    exp = AutoCallbackExperiment(
        model=model,
        loaders=loaders,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        verbose=True,
        check_run=True,
        valid_loader="train",
    )
    _test_callbacks(test_callbacks, exp)


def test_infer_defaults():
    """Docs? Contribution is welcome."""
    test_callbacks = OrderedDict([("_exception", ExceptionCallback)])

    model = torch.nn.Linear(10, 10)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10)
    dataset = torch.utils.data.Dataset()
    dataloader = torch.utils.data.DataLoader(dataset)
    loaders = OrderedDict()
    loaders["train"] = dataloader

    exp = AutoCallbackExperiment(
        model=model,
        loaders=loaders,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        stage="infer",
    )
    _test_callbacks(test_callbacks, exp, "infer")


def test_infer_all():
    """Docs? Contribution is welcome."""
    test_callbacks = OrderedDict(
        [
            ("_verbose", VerboseLogger),
            ("_check", CheckRunCallback),
            ("_exception", ExceptionCallback),
        ]
    )

    model = torch.nn.Linear(10, 10)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10)
    dataset = torch.utils.data.Dataset()
    dataloader = torch.utils.data.DataLoader(dataset)
    loaders = OrderedDict()
    loaders["train"] = dataloader

    exp = AutoCallbackExperiment(
        model=model,
        loaders=loaders,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        verbose=True,
        check_run=True,
        stage="infer",
    )
    _test_callbacks(test_callbacks, exp, "infer")


def test_hparams():
    """
    Test for hparam property of experiment.
    Check if lr, batch_size, optimizer name is in hparams dict
    """
    model = torch.nn.Linear(10, 10)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10)
    dataset = torch.utils.data.Dataset()
    dataloader = torch.utils.data.DataLoader(dataset)
    loaders = OrderedDict()
    loaders["train"] = dataloader

    exp = AutoCallbackExperiment(
        model=model,
        loaders=loaders,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        verbose=True,
        check_run=True,
        stage="infer",
    )
    hparams = exp.hparams

    assert hparams is not None
    assert hparams["lr"] == 1e-3
    assert hparams["train_batch_size"] == 1
    assert hparams["optimizer"] == "Adam"
