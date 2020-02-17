from collections import OrderedDict

import pytest

import torch

from catalyst.dl import registry
from catalyst.dl.callbacks import (
    CheckpointCallback, ConsoleLogger, CriterionCallback, OptimizerCallback,
    PhaseWrapperCallback, RaiseExceptionCallback, TensorboardLogger
)
from catalyst.dl.experiment.config import ConfigExperiment

DEFAULT_MINIMAL_CONFIG = {
    "model_params": {
        "model": "SomeModel"
    },
    "stages": {
        "data_params": {
            "num_workers": 0
        },
        "train": {}
    }
}


DEFAULT_CALLBACKS = OrderedDict([
    ("_criterion", CriterionCallback),
    ("_optimizer", OptimizerCallback),
    ("_saver", CheckpointCallback),
    ("console", ConsoleLogger),
    ("tensorboard", TensorboardLogger),
    ("exception", RaiseExceptionCallback)])


class SomeModel(torch.nn.Module):
    """
    Dummy test torch model
    """
    pass


registry.MODELS.add(SomeModel)


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
    assert exp.logdir is None
    assert exp.stages == ["train"]
    assert exp.distributed_params == {}
    assert exp.monitoring_params == {}
    assert exp.get_state_params("train") == {
        "logdir": None,
    }
    assert isinstance(exp.get_model("train"), SomeModel)
    assert exp.get_criterion("train") is None
    assert exp.get_optimizer("train", SomeModel()) is None
    assert exp.get_scheduler("train", None) is None
    assert exp.get_callbacks("train").keys() == DEFAULT_CALLBACKS.keys()
    cbs = zip(exp.get_callbacks("train").values(), DEFAULT_CALLBACKS.values())
    for c1, klass in cbs:
        assert isinstance(c1, klass)


def test_when_callback_defined():
    """
    There should be no default callback of same kind if there is user defined
    already.
    """
    config = DEFAULT_MINIMAL_CONFIG
    config["stages"]["callbacks_params"] = {
        "my_criterion": {
            "callback": "CriterionCallback"
        }
    }
    exp = ConfigExperiment(config=config)

    assert "_criterion" not in exp.get_callbacks("train").keys()
    assert "my_criterion" in exp.get_callbacks("train").keys()


def test_when_callback_wrapped():
    """
    There should be no default callback of same kind of user defined wrapped
    callback.
    """
    config = DEFAULT_MINIMAL_CONFIG
    config["stages"]["callbacks_params"] = {
        "my_wrapped_criterion": {
            "_wrapper": {
                "callback": "PhaseBatchWrapperCallback",
                "active_phases": [1]
            },
            "callback": "CriterionCallback"
        }
    }
    exp = ConfigExperiment(config=config)

    assert "_criterion" not in exp.get_callbacks("train").keys()
    callback = exp.get_callbacks("train")["my_wrapped_criterion"]
    assert isinstance(callback, PhaseWrapperCallback)


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
