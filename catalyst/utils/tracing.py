from typing import Tuple, Union
import inspect
import logging

import torch
from torch import jit, nn

from catalyst.tools.forward_wrapper import ModelForwardWrapper
from catalyst.typing import Model
from catalyst.utils.misc import get_fn_argsnames
from catalyst.utils.torch import set_requires_grad

logger = logging.getLogger(__name__)


def trace_model(
    model: Model,
    batch: Union[Tuple[torch.Tensor], torch.Tensor],
    method_name: str = "forward",
    predict_params: dict = None,
) -> jit.ScriptModule:
    """Traces model using runner and batch.

    Args:
        model: Model to trace
        batch: Batch to trace the model
        method_name: Model's method name that will be
            used as entrypoint during tracing

    Example:
        .. code-block:: python

           import torch

           from catalyst.utils import trace_model

           class LinModel(torch.nn.Module):
               def __init__(self):
                   super().__init__()
                   self.lin1 = torch.nn.Linear(10, 10)
                   self.lin2 = torch.nn.Linear(2, 10)

               def forward(self, inp_1, inp_2):
                   return self.lin1(inp_1), self.lin2(inp_2)
            
               def first_only(self, inp_1):
                   return self.lin1(inp_1)
           
           lin_model = LinModel()
           traced_model = trace_model(
               lin_model, batch=torch.randn(1, 10), method_name="first_only"
           )


    Returns:
        jit.ScriptModule: Traced model
    """

    predict_params = predict_params or {}

    wrapped_model = ModelForwardWrapper(model=model, method_name=method_name)
    traced = jit.trace(wrapped_model, example_inputs=batch)
    return traced


__all__ = [
    "trace_model",
]
