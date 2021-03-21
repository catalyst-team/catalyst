from typing import Union
from pathlib import Path

import torch
from torch import Tensor

from catalyst.core import Callback, CallbackNode, CallbackOrder
from catalyst.utils import trace_model


class TracingCallback(Callback):
    def __init__(
        self,
        batch: Tensor = None,
        method_name: str = "forward",
        logdir: Union[str, Path] = None,
        filename: str = "traced_model",
    ):
        super().__init__(order=CallbackOrder.ExternalExtra, node=CallbackNode.Master)
        if logdir is not None:
            self.filename = Path(logdir) / filename
        else:
            self.filename = filename
        self.method_name = method_name
        self.batch = batch

    def on_stage_end(self, runner: "IRunner") -> None:
        model = runner.model
        batch = self.batch or next(iter(runner.loaders["train"]))
        traced_model = trace_model(model=model, batch=batch, method_name=self.method_name)
        torch.jit.save(traced_model, self.filename)


__all__ = ["TracingCallback"]
