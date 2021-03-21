from typing import Union, Iterable, List, Dict, TYPE_CHECKING
from pathlib import Path

from torch import Tensor

from catalyst.utils import onnx_export
from catalyst.core import CallbackOrder, CallbackNode, Callback

if TYPE_CHECKING:
    from catalyst.core import IRunner


class OnnxCallback(Callback):
    def __init__(
        self,
        filename: str = "onnx.py",
        logdir: Union[str, Path] = None,
        batch: Tensor = None,
        method_name: str = "forward",
        input_names: Iterable = None,
        output_names: List[str] = None,
        dynamic_axes: Union[Dict[str, int], Dict[str, Dict[str, int]]] = None,
        opset_version: int = 9,
        do_constant_folding: bool = False,
        verbose: bool = False,
    ):
        super().__init__(order=CallbackOrder.ExternalExtra, node=CallbackNode.Master)
        if self.logdir is not None:
            self.filename = Path(logdir) / filename
        else:
            self.filename = filename

        self.method_name = method_name
        self.input_names = input_names
        self.output_names = output_names
        self.dynamic_axes = dynamic_axes
        self.opset_version = opset_version
        self.do_constant_folding = do_constant_folding
        self.verbose = verbose
        self.batch = batch

    def on_stage_end(self, runner: "IRunner") -> None:
        model = runner.model.cpu()
        batch = self.batch or next(iter(runner.loaders["train"]))
        onnx_export(
            model=model,
            file=self.filename,
            batch=batch,
            method_name=self.method_name,
            input_names=self.input_names,
            output_names=self.output_names,
            dynamic_axes=self.dynamic_axes,
            opset_version=self.opset_version,
            do_constant_folding=self.do_constant_folding,
            verbose=self.verbose
        )


__all__ = ["OnnxCallback"]