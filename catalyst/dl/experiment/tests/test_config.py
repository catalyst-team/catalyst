from collections import OrderedDict

import pytest

import torch

from catalyst.dl import (
    CheckpointCallback,
    ConsoleLogger,
    ExceptionCallback,
    MetricManagerCallback,
    registry,
    TensorboardLogger,
    ValidationManagerCallback,
)
from catalyst.dl.experiment.config import ConfigExperiment

DEFAULT_MINIMAL_CONFIG = {
    "model_params": {"model": "SomeModel"},
    "stages": {"data_params": {"num_workers": 0}, "train": {}},
    "args": {"logdir": "./logdir"},
}

DEFAULT_CALLBACKS = OrderedDict(
    [
        ("_metrics", MetricManagerCallback),
        ("_validation", ValidationManagerCallback),
        ("_saver", CheckpointCallback),
        ("_console", ConsoleLogger),
        ("_tensorboard", TensorboardLogger),
        ("_exception", ExceptionCallback),
    ]
)


class SomeModel(torch.nn.Module):
    """Dummy test torch model."""

    pass


registry.MODELS.add(SomeModel)


def _test_callbacks(test_callbacks, exp, stage="train"):
    exp_callbacks = exp.get_callbacks(stage)
    exp_callbacks = OrderedDict(
        sorted(exp_callbacks.items(), key=lambda t: t[0])
    )
    test_callbacks = OrderedDict(
        sorted(test_callbacks.items(), key=lambda t: t[0])
    )
    print(test_callbacks.keys())
    print(exp_callbacks.keys())

    assert exp_callbacks.keys() == test_callbacks.keys()
    cbs = zip(exp_callbacks.values(), test_callbacks.values())
    for callback, klass in cbs:
        assert isinstance(callback, klass)


def test_defaults():
    """
    Test on ConfigExperiment defaults.
    It's pretty similar to BaseExperiment's test
    but the thing is that those two are very different classes and
    inherit from different parent classes.
    Also very important to check which callbacks are added by default
    """
    exp = ConfigExperiment(config=DEFAULT_MINIMAL_CONFIG)

    assert exp.initial_seed == 42
    assert exp.logdir == "./logdir"
    assert exp.stages == ["train"]
    assert exp.distributed_params == {}
    assert exp.get_state_params("train") == {
        "logdir": "./logdir",
    }
    assert isinstance(exp.get_model("train"), SomeModel)
    assert exp.get_criterion("train") is None
    assert exp.get_optimizer("train", SomeModel()) is None
    assert exp.get_scheduler("train", None) is None

    _test_callbacks(DEFAULT_CALLBACKS, exp)


def test_when_callback_defined():
    """
    There should be no default callback of same kind if there is user defined
    already.
    """
    config = DEFAULT_MINIMAL_CONFIG.copy()
    config["stages"]["callbacks_params"] = {
        "my_criterion": {"callback": "CriterionCallback"}
    }
    exp = ConfigExperiment(config=config)

    assert "_criterion" not in exp.get_callbacks("train").keys()
    assert "my_criterion" in exp.get_callbacks("train").keys()


def test_not_implemented_datasets():
    """
    Test on ``get_datasets`` method, which should be implememnted by user.
    Method ``get_loaders`` will call ``get_dataset``.
    """
    exp = ConfigExperiment(config=DEFAULT_MINIMAL_CONFIG)

    with pytest.raises(NotImplementedError):
        exp.get_loaders("train")
    with pytest.raises(NotImplementedError):
        exp.get_datasets("train")
