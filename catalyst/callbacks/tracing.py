from typing import Union, TYPE_CHECKING
from pathlib import Path

import torch
from torch import Tensor

from catalyst.core import Callback, CallbackNode, CallbackOrder
from catalyst.utils import trace_model

if TYPE_CHECKING:
    from catalyst.core import IRunner


class TracingCallback(Callback):
    """
    Callback for model tracing.

    Args:
        logdir: path to folder for saving
        filename: filename
        batch: input tensor for model. If None will take batch from train loader.
        method_name: Model's method name that will be used as entrypoint during tracing
    """
    def __init__(
        self,
        logdir: Union[str, Path] = None,
        filename: str = "traced_model.pth",
        batch: Tensor = None,
        method_name: str = "forward",
    ):
        """
        Callback for model tracing.

        Args:
            logdir: path to folder for saving
            filename: filename
            batch: input tensor for model. If None will take batch from train loader.
            method_name: Model's method name that will be used as entrypoint during tracing
        """
        super().__init__(order=CallbackOrder.ExternalExtra, node=CallbackNode.Master)
        if logdir is not None:
            self.filename = Path(logdir) / filename
        else:
            self.filename = filename
        self.method_name = method_name
        self.batch = batch

    def on_stage_end(self, runner: "IRunner") -> None:
        """
        On stage end action.

        Args:
            runner: runner for experiment
        """
        model = runner.model
        batch = self.batch or next(iter(runner.loaders["train"]))
        traced_model = trace_model(model=model, batch=batch, method_name=self.method_name)
        torch.jit.save(traced_model, self.filename)


__all__ = ["TracingCallback"]
