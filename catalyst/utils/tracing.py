from typing import Any, Callable, List
import inspect
import logging

from torch import jit, nn

from catalyst.tools.forward_pass import ForwardOverrideModel
from catalyst.typing import Device, Model
from catalyst.utils.misc import get_fn_argsnames
from catalyst.utils.torch import set_requires_grad

logger = logging.getLogger(__name__)


def _get_input_argnames(fn: Callable[..., Any], exclude: List[str] = None) -> List[str]:
    """
    Function to get input argument names of function.

    Args:
        fn (Callable[..., Any]): Function to get argument names from
        exclude: List of string of names to exclude

    Returns:
        List[str]: List of input argument names
    """
    argspec = inspect.getfullargspec(fn)
    assert argspec.varargs is None and argspec.varkw is None, "not supported by PyTorch"

    return get_fn_argsnames(fn, exclude=exclude)


class _TracingModelWrapper(nn.Module):
    """Wrapper that traces model with batch instead of calling it.

    (Workaround, to use native model batch handler)
    """

    def __init__(self, model, method_name):
        super().__init__()
        self.model = model
        self.method_name = method_name
        self.tracing_result: jit.ScriptModule

    def __call__(self, *args, **kwargs):
        method_model = ForwardOverrideModel(self.model, self.method_name)

        try:
            assert len(args) == 0, "only KV support implemented"

            fn = getattr(self.model, self.method_name)
            method_argnames = _get_input_argnames(fn=fn, exclude=["self"])
            method_input = tuple(kwargs[name] for name in method_argnames)

            self.tracing_result = jit.trace(method_model, method_input)
        except Exception:
            # for backward compatibility
            self.tracing_result = jit.trace(method_model, *args, **kwargs)
        output = self.model.forward(*args, **kwargs)

        return output


def trace_model(
    model: Model,
    predict_fn: Callable,
    batch,
    method_name: str = "forward",
    mode: str = "eval",
    requires_grad: bool = False,
    predict_params: dict = None,
) -> jit.ScriptModule:
    """Traces model using runner and batch.

    Args:
        model: Model to trace
        predict_fn: Function to run prediction with the model provided,
            takes model, inputs parameters
        batch: Batch to trace the model
        method_name: Model's method name that will be
            used as entrypoint during tracing
        mode: Mode for model to trace (``train`` or ``eval``)
        requires_grad: Flag to use grads
        predict_params: additional parameters for model forward

    Returns:
        jit.ScriptModule: Traced model

    Raises:
        ValueError: if both batch and predict_fn must be specified or
          mode is not in 'eval' or 'train'.
    """
    if batch is None or predict_fn is None:
        raise ValueError("Both batch and predict_fn must be specified.")

    if mode not in ["train", "eval"]:
        raise ValueError(f"Unknown mode '{mode}'. Must be 'eval' or 'train'")

    predict_params = predict_params or {}

    wrapped_model = _TracingModelWrapper(model, method_name)

    getattr(model, mode)()
    set_requires_grad(model, requires_grad=requires_grad)

    predict_fn(wrapped_model, batch, **predict_params)

    return wrapped_model.tracing_result


__all__ = [
    "trace_model",
]
