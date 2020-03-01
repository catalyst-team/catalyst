from collections import OrderedDict

import torch

from catalyst.dl import (
    CheckpointCallback, ConsoleLogger, CriterionCallback, ExceptionCallback,
    OptimizerCallback, SchedulerCallback, TensorboardLogger
)
from catalyst.dl.experiment.supervised import SupervisedExperiment

DEFAULT_CALLBACKS = OrderedDict(
    [
        ("_saver", CheckpointCallback), ("console", ConsoleLogger),
        ("tensorboard", TensorboardLogger), ("exception", ExceptionCallback)
    ]
)


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

    exp = SupervisedExperiment(model=model, loaders=loaders)

    assert exp.get_callbacks("train").keys() == DEFAULT_CALLBACKS.keys()
    cbs = zip(exp.get_callbacks("train").values(), DEFAULT_CALLBACKS.values())
    for callback, klass in cbs:
        assert isinstance(callback, klass)


def test_full():
    FULL_CALLBACKS = DEFAULT_CALLBACKS.copy()
    FULL_CALLBACKS["_criterion"] = CriterionCallback
    FULL_CALLBACKS["_optimizer"] = OptimizerCallback
    FULL_CALLBACKS["_scheduler"] = SchedulerCallback

    model = torch.nn.Linear(10, 10)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10)
    dataset = torch.utils.data.Dataset()
    dataloader = torch.utils.data.DataLoader(dataset)
    loaders = OrderedDict()
    loaders["train"] = dataloader

    exp = SupervisedExperiment(
        model=model,
        loaders=loaders,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
    )

    exp_callbacks = exp.get_callbacks("train")
    exp_callbacks = OrderedDict(
        sorted(exp_callbacks.items(), key=lambda t: t[0])
    )
    FULL_CALLBACKS = OrderedDict(
        sorted(FULL_CALLBACKS.items(), key=lambda t: t[0])
    )

    assert exp_callbacks.keys() == FULL_CALLBACKS.keys()
    cbs = zip(exp_callbacks.values(), FULL_CALLBACKS.values())
    for callback, klass in cbs:
        assert isinstance(callback, klass)
