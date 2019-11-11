import torch
from torch import nn
from torch.jit import ScriptModule

from catalyst import utils
from catalyst.dl.core import Runner
from catalyst.utils.typing import Device, Model


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

        self.tracing_result = torch.jit.trace(
            method_model, *args, **kwargs
        )


def trace_model(
    model: Model,
    runner: Runner,
    batch=None,
    method_name: str = "forward",
    mode: str = "eval",
    requires_grad: bool = False,
    opt_level: str = None,
    device: Device = "cpu",
) -> ScriptModule:
    """
    Traces model using it's native experiment and runner.

    Args:
        model: Model to trace
        batch: Batch to trace the model
        runner: Model's native runner that was used to train model
        experiment: Native experiment that was used to train model
        method_name (str): Model's method name that will be
            used as entrypoint during tracing
        mode (str): Mode for model to trace (``train`` or ``eval``)
        requires_grad (bool): Flag to use grads
        opt_level (str): AMP FP16 init level, optional
        device (str): Torch device

    Returns:
        (ScriptModule): Traced model
    """
    if batch is None or runner is None:
        raise ValueError("Both batch and runner must be specified.")

    if mode not in ["train", "eval"]:
        raise ValueError(f"Unknown mode '{mode}'. Must be 'eval' or 'train'")

    tracer = _TracingModelWrapper(model, method_name)
    if opt_level is not None:
        utils.assert_fp16_available()
        # If traced in AMP we need to initialize the model before calling
        # the jit
        # https://github.com/NVIDIA/apex/issues/303#issuecomment-493142950
        from apex import amp
        model = model.to(device)
        model = amp.initialize(model, optimizers=None, opt_level=opt_level)
        # TODO: remove `check_trace=False`
        # after fixing this bug https://github.com/pytorch/pytorch/issues/23993
        params = dict(check_trace=False)
    else:
        params = dict()

    getattr(model, mode)()
    utils.set_requires_grad(model, requires_grad=requires_grad)

    _runner_model, _runner_device = runner.get_model_device()

    runner.set_model_device(tracer.to(device), device)
    runner.predict_batch(batch, **params)
    result: ScriptModule = tracer.tracing_result

    runner.set_model_device(_runner_model, _runner_device)
    return result


def get_trace_name(
    method_name: str,
    mode: str = "eval",
    requires_grad: bool = False,
    opt_level: str = None,
    additional_string: str = None,
):
    """
    Creates a file name for the traced model.

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


__all__ = [
    "trace_model",
    "get_trace_name"
]
