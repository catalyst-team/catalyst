from typing import Type

import torch
from torch import nn
from torch.jit import ScriptModule

from catalyst.dl.core import Experiment, Runner
from catalyst import utils


class _ForwardOverrideModel(nn.Module):
    """
    Model that calls specified method instead of forward

    (Workaround, single method tracing is not supported)
    """

    def __init__(self, model, method_name):
        super().__init__()
        self.model = model
        self.method = method_name

    def forward(self, *args, **kwargs):
        return getattr(self.model, self.method)(*args, **kwargs)


class _TracingModelWrapper(nn.Module):
    """
    Wrapper that traces model with batch instead of calling it

    (Workaround, to use native model batch handler)
    """

    def __init__(self, model, method_name):
        super().__init__()
        self.method_name = method_name
        self.model = model
        self.tracing_result: ScriptModule

    def __call__(self, *args, **kwargs):
        method_model = _ForwardOverrideModel(self.model, self.method_name)

        self.tracing_result = \
            torch.jit.trace(
                method_model,
                *args, **kwargs
            )


def _get_native_batch(experiment: Experiment, stage: str):
    """Returns dataset from first loader provided by experiment"""
    loaders = experiment.get_loaders(stage)
    assert loaders, \
        "Experiment must have at least one loader to support tracing"
    # Take first loader
    loader = next(iter(loaders.values()))
    dataset = loader.dataset
    collate_fn = loader.collate_fn

    sample = collate_fn([dataset[0]])

    return sample


def trace_model(
    model: nn.Module,
    experiment: Experiment,
    runner_type: Type[Runner],
    method_name: str = "forward",
    mode: str = "eval",
    requires_grad: bool = False,
) -> ScriptModule:
    """
    Traces model using it's native experiment and runner.

    Args:
        model: Model to trace
        experiment: Native experiment that was used to train model
        runner_type: Model's native runner that was used to train model
        method_name (str): Model's method name that will be
            used as entrypoint during tracing
        mode (str): Mode for model to trace (``train`` or ``eval``)
        requires_grad (bool): Flag to use grads

    Returns:
        Traced model ScriptModule
    """

    if mode not in ["train", "eval"]:
        raise ValueError(f"Unknown mode '{mode}'. Must be 'eval' or 'train'")

    getattr(model, mode)()
    utils.set_requires_grad(model, requires_grad=requires_grad)

    tracer = _TracingModelWrapper(model, method_name)
    runner: Runner = runner_type(tracer.cpu(), torch.device("cpu"))

    stage = list(experiment.stages)[0]
    batch = _get_native_batch(experiment, stage)
    batch = runner._batch2device(batch, device=runner.device)

    runner.predict_batch(batch)

    return tracer.tracing_result


__all__ = ["trace_model"]
