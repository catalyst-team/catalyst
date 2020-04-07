from typing import TYPE_CHECKING, Union
import inspect
from pathlib import Path

import torch
from torch import nn
from torch.jit import ScriptModule

from catalyst.utils import (
    assert_fp16_available,
    get_fn_argsnames,
    set_requires_grad,
)
from catalyst.utils.tools.typing import Device, Model

if TYPE_CHECKING:
    from catalyst.dl import Runner  # noqa: F401


class _ForwardOverrideModel(nn.Module):
    """Model that calls specified method instead of forward.

    (Workaround, single method tracing is not supported)
    """

    def __init__(self, model, method_name):
        super().__init__()
        self.model = model
        self.method_name = method_name

    def forward(self, *args, **kwargs):
        return getattr(self.model, self.method_name)(*args, **kwargs)


class _TracingModelWrapper(nn.Module):
    """Wrapper that traces model with batch instead of calling it.

    (Workaround, to use native model batch handler)
    """

    def __init__(self, model, method_name):
        super().__init__()
        self.model = model
        self.method_name = method_name
        self.tracing_result: ScriptModule

    def __call__(self, *args, **kwargs):
        method_model = _ForwardOverrideModel(self.model, self.method_name)

        try:
            assert len(args) == 0, "only KV support implemented"

            fn = getattr(self.model, self.method_name)
            argspec = inspect.getfullargspec(fn)
            assert (
                argspec.varargs is None and argspec.varkw is None
            ), "not supported by PyTorch tracing"

            method_argnames = get_fn_argsnames(fn, exclude=["self"])
            method_input = tuple(kwargs[name] for name in method_argnames)

            self.tracing_result = torch.jit.trace(method_model, method_input)
        except Exception:
            # for backward compatibility
            self.tracing_result = torch.jit.trace(
                method_model, *args, **kwargs
            )
        output = self.model.forward(*args, **kwargs)

        return output


def trace_model(
    model: Model,
    runner: "Runner",
    batch=None,
    method_name: str = "forward",
    mode: str = "eval",
    requires_grad: bool = False,
    opt_level: str = None,
    device: Device = "cpu",
    predict_params: dict = None,
) -> ScriptModule:
    """Traces model using runner and batch.

    Args:
        model: Model to trace
        runner: Model's native runner that was used to train model
        batch: Batch to trace the model
        method_name (str): Model's method name that will be
            used as entrypoint during tracing
        mode (str): Mode for model to trace (``train`` or ``eval``)
        requires_grad (bool): Flag to use grads
        opt_level (str): Apex FP16 init level, optional
        device (str): Torch device
        predict_params (dict): additional parameters for model forward

    Returns:
        (ScriptModule): Traced model
    """
    if batch is None or runner is None:
        raise ValueError("Both batch and runner must be specified.")

    if mode not in ["train", "eval"]:
        raise ValueError(f"Unknown mode '{mode}'. Must be 'eval' or 'train'")

    predict_params = predict_params or {}

    tracer = _TracingModelWrapper(model, method_name)
    if opt_level is not None:
        assert_fp16_available()
        # If traced in AMP we need to initialize the model before calling
        # the jit
        # https://github.com/NVIDIA/apex/issues/303#issuecomment-493142950
        from apex import amp

        model = model.to(device)
        model = amp.initialize(model, optimizers=None, opt_level=opt_level)
        # TODO: remove `check_trace=False`
        # after fixing this bug https://github.com/pytorch/pytorch/issues/23993
        params = {**predict_params, "check_trace": False}
    else:
        params = predict_params

    getattr(model, mode)()
    set_requires_grad(model, requires_grad=requires_grad)

    _runner_model, _runner_device = runner.model, runner.device

    runner.model, runner.device = tracer, device
    runner.predict_batch(batch, **params)
    result: ScriptModule = tracer.tracing_result

    runner.model, runner.device = _runner_model, _runner_device
    return result


def get_trace_name(
    method_name: str,
    mode: str = "eval",
    requires_grad: bool = False,
    opt_level: str = None,
    additional_string: str = None,
):
    """Creates a file name for the traced model.

    Args:
        method_name (str): model's method name
        mode (str): ``train`` or ``eval``
        requires_grad (bool): flag if model was traced with gradients
        opt_level (str): opt_level if model was traced in FP16
        additional_string (str): any additional information
    """
    file_name = f"traced"
    if additional_string is not None:
        file_name += f"-{additional_string}"

    file_name += f"-{method_name}"
    if mode == "train":
        file_name += "-in_train"

    if requires_grad:
        file_name += f"-with_grad"

    if opt_level is not None:
        file_name += f"-opt_{opt_level}"

    file_name += ".pth"

    return file_name


def load_traced_model(
    model_path: Union[str, Path],
    device: Device = "cpu",
    opt_level: str = None,
) -> ScriptModule:
    """Loads a traced model.

    Args:
        model_path: Path to traced model
        device (str): Torch device
        opt_level (str): Apex FP16 init level, optional

    Returns:
        (ScriptModule): Traced model
    """
    # jit.load dont work with pathlib.Path
    model_path = str(model_path)

    if opt_level is not None:
        device = "cuda"

    model = torch.jit.load(model_path, map_location=device)

    if opt_level is not None:
        assert_fp16_available()
        from apex import amp

        model = amp.initialize(model, optimizers=None, opt_level=opt_level)

    return model


__all__ = [
    "trace_model",
    "get_trace_name",
    "load_traced_model",
]
