from collections import OrderedDict

import torch

from catalyst.dl import (
    ConsoleLogger,
    ExceptionCallback,
    MetricManagerCallback,
    OptimizerCallback,
    PhaseManagerCallback,
    PhaseWrapperCallback,
    TensorboardLogger,
    ValidationManagerCallback,
)
from catalyst.dl.experiment.gan import GanExperiment

DEFAULT_CALLBACKS = OrderedDict(
    [
        ("phase_manager", PhaseManagerCallback),
        ("_metrics", MetricManagerCallback),
        ("_validation", ValidationManagerCallback),
        ("_console", ConsoleLogger),
        ("_exception", ExceptionCallback),
    ]
)


def test_defaults():
    """
    Test on defaults for GanExperiment class, which is child class of
    BaseExperiment.  That's why we check only default callbacks functionality
    here.
    """
    model = torch.nn.Module()
    dataset = torch.utils.data.Dataset()
    dataloader = torch.utils.data.DataLoader(dataset)
    loaders = OrderedDict()
    loaders["train"] = dataloader
    state_kwargs = {
        "discriminator_train_phase": "discriminator_train",
        "discriminator_train_num": 1,
        "generator_train_phase": "generator_train",
        "generator_train_num": 5,
    }
    exp = GanExperiment(
        model=model,
        loaders=loaders,
        state_kwargs=state_kwargs,
        valid_loader="train",
    )

    assert exp.get_callbacks("train").keys() == DEFAULT_CALLBACKS.keys()
    cbs = zip(exp.get_callbacks("train").values(), DEFAULT_CALLBACKS.values())
    for callback, klass in cbs:
        assert isinstance(callback, klass)


def test_callback_wrapping():
    """Test on callback wrapping for GanExperiment class."""
    model = torch.nn.Module()
    dataset = torch.utils.data.Dataset()
    dataloader = torch.utils.data.DataLoader(dataset)
    loaders = OrderedDict()
    loaders["train"] = dataloader
    # Prepare callbacks and state kwargs
    discriminator_loss_key = "loss_d"
    generator_loss_key = "loss_g"
    discriminator_key = "discriminator"
    generator_key = "generator"
    input_callbacks = OrderedDict(
        {
            "optim_d": OptimizerCallback(
                loss_key=discriminator_loss_key,
                optimizer_key=discriminator_key,
            ),
            "optim_g": OptimizerCallback(
                loss_key=generator_loss_key, optimizer_key=generator_key
            ),
            "tensorboard": TensorboardLogger(),
        }
    )
    state_kwargs = {
        "discriminator_train_phase": "discriminator_train",
        "discriminator_train_num": 1,
        "generator_train_phase": "generator_train",
        "generator_train_num": 5,
    }
    discriminator_callbacks = ["optim_d"]
    generator_callbacks = ["optim_g"]
    phase2callbacks = {
        state_kwargs["discriminator_train_phase"]: discriminator_callbacks,
        state_kwargs["generator_train_phase"]: generator_callbacks,
    }

    exp = GanExperiment(
        model=model,
        loaders=loaders,
        callbacks=input_callbacks,
        state_kwargs=state_kwargs,
        phase2callbacks=phase2callbacks,
        valid_loader="train",
    )

    callbacks = exp.get_callbacks("train")
    assert "optim_d" in callbacks.keys()
    assert "optim_g" in callbacks.keys()
    assert "tensorboard" in callbacks.keys()
    assert isinstance(callbacks["phase_manager"], PhaseManagerCallback)
    assert isinstance(callbacks["optim_d"], PhaseWrapperCallback)
    assert isinstance(callbacks["optim_g"], PhaseWrapperCallback)
    assert isinstance(callbacks["tensorboard"], TensorboardLogger)
