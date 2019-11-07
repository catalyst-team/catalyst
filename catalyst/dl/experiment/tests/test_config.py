from collections import OrderedDict

import pytest

import torch

from catalyst.dl import registry
from catalyst.dl.callbacks import (
    CheckpointCallback, ConsoleLogger, CriterionCallback, OptimizerCallback,
    RaiseExceptionCallback, TensorboardLogger
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
    pass


registry.MODELS.add(SomeModel)


def test_defaults():
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


def test_not_implemented_datasets():
    exp = ConfigExperiment(config=DEFAULT_MINIMAL_CONFIG)

    with pytest.raises(NotImplementedError):
        exp.get_loaders("train")
    with pytest.raises(NotImplementedError):
        exp.get_datasets("train")
