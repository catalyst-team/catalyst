from collections import OrderedDict

import torch

from catalyst.dl.callbacks import (
    CheckpointCallback, ConsoleLogger, CriterionCallback, OptimizerCallback,
    RaiseExceptionCallback, TensorboardLogger
)
from catalyst.dl.experiment.supervised import SupervisedExperiment

DEFAULT_CALLBACKS = OrderedDict(
    [
        ("_criterion", CriterionCallback), ("_optimizer", OptimizerCallback),
        ("_saver", CheckpointCallback), ("console", ConsoleLogger),
        ("tensorboard", TensorboardLogger),
        ("exception", RaiseExceptionCallback)
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
