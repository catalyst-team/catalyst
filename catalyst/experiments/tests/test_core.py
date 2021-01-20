from collections import OrderedDict

import torch

from catalyst.callbacks import (
    ConsoleLogger,
    ExceptionCallback,
    MetricManagerCallback,
    ValidationManagerCallback,
)
from catalyst.experiments import Experiment


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
    Test on defaults for BaseExperiment. It will be useful if we decide to
    change anything in those values as it could make breaking change.
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

    exp = Experiment(model=model, loaders=loaders, valid_loader="train")

    assert exp.seed == 42
    assert exp.logdir is None
    assert exp.stages == ["train"]
    assert exp.engine_params == {}
    assert exp.get_stage_params("") == {
        "logdir": None,
        "num_epochs": 1,
        "valid_loader": "train",
        "main_metric": "loss",
        "verbose": False,
        "minimize_metric": True,
        "checkpoint_data": {},
    }
    assert exp.get_model("") == model
    assert exp.get_criterion("") is None
    assert exp.get_optimizer("", model) is None
    assert exp.get_scheduler("") is None
    _test_callbacks(test_callbacks, exp)
    assert exp.get_loaders("") == loaders
