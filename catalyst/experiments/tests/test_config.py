from collections import OrderedDict

import pytest
import torch

from catalyst import registry
from catalyst.callbacks import (
    CheckpointCallback,
    ConsoleLogger,
    CriterionCallback,
    ExceptionCallback,
    MetricManagerCallback,
    OptimizerCallback,
    SchedulerCallback,
    TensorboardLogger,
    ValidationManagerCallback,
)
from catalyst.experiments import ConfigExperiment

DEFAULT_MINIMAL_CONFIG = {  # noqa: WPS407
    "model_params": {"_target_": "SomeModel"},
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


class SomeOptimizer(torch.nn.Module):
    """Dummy test torch optimizer."""

    def __init__(self, **kwargs):
        """Dummy optimizer"""
        super().__init__()


class SomeScheduler(torch.nn.Module):
    """Dummy test torch scheduler."""

    def __init__(self, **kwargs):
        """Dummy scheduler"""
        super().__init__()


registry.REGISTRY.add(SomeModel)
registry.REGISTRY.add(SomeOptimizer)
registry.REGISTRY.add(SomeScheduler)


def _test_callbacks(test_callbacks, exp, stage="train"):
    exp_callbacks = exp.get_callbacks(stage)
    exp_callbacks = OrderedDict(sorted(exp_callbacks.items(), key=lambda t: t[0]))
    test_callbacks = OrderedDict(sorted(test_callbacks.items(), key=lambda t: t[0]))
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
    exp = ConfigExperiment(config=DEFAULT_MINIMAL_CONFIG.copy())

    assert exp.seed == 42
    assert exp.logdir == "./logdir"
    assert exp.stages == ["train"]
    assert exp.engine_params == {}
    assert exp.get_stage_params("train") == {
        "logdir": "./logdir",
    }
    assert isinstance(exp.get_model("train"), SomeModel)
    assert exp.get_criterion("train") is None
    assert exp.get_optimizer("train", SomeModel()) is None
    assert exp.get_scheduler("train", None) is None

    _test_callbacks(DEFAULT_CALLBACKS, exp)


def test_defaults_criterion_optimizer_scheduler():
    """
    Test on ConfigExperiment defaults.
    when {criterion, optimizer, scheduler}_params are specified
    the respective callback should be generated automatically
    """
    callbacks = DEFAULT_CALLBACKS.copy()
    callbacks["_criterion"] = CriterionCallback
    callbacks["_optimizer"] = OptimizerCallback
    callbacks["_scheduler"] = SchedulerCallback

    config = DEFAULT_MINIMAL_CONFIG.copy()
    config["stages"]["criterion_params"] = {"_target_": "BCEWithLogitsLoss"}
    config["stages"]["optimizer_params"] = {"_target_": "SomeOptimizer"}
    config["stages"]["scheduler_params"] = {"_target_": "SomeScheduler"}
    exp = ConfigExperiment(config=config)

    assert exp.seed == 42
    assert exp.logdir == "./logdir"
    assert exp.stages == ["train"]
    assert exp.engine_params == {}
    assert exp.get_stage_params("train") == {
        "logdir": "./logdir",
    }
    assert isinstance(exp.get_model("train"), SomeModel)
    assert exp.get_criterion("train") is not None
    assert exp.get_optimizer("train", SomeModel()) is not None
    assert exp.get_scheduler("train", None) is not None

    _test_callbacks(callbacks, exp)


def test_when_callback_defined():
    """
    There should be no default callback of same kind if there is user defined
    already.
    """
    callbacks = DEFAULT_CALLBACKS.copy()
    callbacks["my_criterion"] = CriterionCallback
    callbacks["my_optimizer"] = OptimizerCallback
    callbacks["my_scheduler"] = SchedulerCallback

    config = DEFAULT_MINIMAL_CONFIG.copy()
    config["stages"]["criterion_params"] = {"_target_": "BCEWithLogitsLoss"}
    config["stages"]["optimizer_params"] = {"_target_": "SomeOptimizer"}
    config["stages"]["scheduler_params"] = {"_target_": "SomeScheduler"}
    config["stages"]["callbacks_params"] = {
        "my_criterion": {"_target_": "CriterionCallback"},
        "my_optimizer": {"_target_": "OptimizerCallback"},
        "my_scheduler": {"_target_": "SchedulerCallback"},
    }
    exp = ConfigExperiment(config=config)
    _test_callbacks(callbacks, exp)


def test_not_implemented_datasets():
    """
    Test on ``get_datasets`` method, which should be implememnted by user.
    Method ``get_loaders`` will call ``get_dataset``.
    """
    exp = ConfigExperiment(config=DEFAULT_MINIMAL_CONFIG.copy())

    with pytest.raises(NotImplementedError):
        exp.get_loaders("train")
    with pytest.raises(NotImplementedError):
        exp.get_datasets("train")
