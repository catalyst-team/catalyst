from typing import Dict, Iterable, List, TYPE_CHECKING, Union
from pathlib import Path

from catalyst.core import Callback, CallbackNode, CallbackOrder
from catalyst.utils import onnx_export

if TYPE_CHECKING:
    from catalyst.core import IRunner


class OnnxCallback(Callback):
    """
    Callback for converting model to onnx runtime.

    Args:
        input_key: input key from ``runner.batch`` to use for onnx export
        logdir: path to folder for saving
        filename: filename
        method_name (str, optional): Forward pass method to be converted. Defaults to "forward".
        input_names (Iterable, optional): name of inputs in graph. Defaults to None.
        output_names (List[str], optional): name of outputs in graph. Defaults to None.
        dynamic_axes (Union[Dict[str, int], Dict[str, Dict[str, int]]], optional): axes
            with dynamic shapes. Defaults to None.
        opset_version (int, optional): Defaults to 9.
        do_constant_folding (bool, optional): If True, the constant-folding optimization
            is applied to the model during export. Defaults to False.
        verbose (bool, default False): if specified, we will print out a debug
            description of the trace being exported.

    Example:
        .. code-block:: python

            import os

            import torch
            from torch import nn
            from torch.utils.data import DataLoader

            from catalyst import dl
            from catalyst.data.transforms import ToTensor
            from catalyst.contrib.datasets import MNIST
            from catalyst.contrib.nn.modules import Flatten

            loaders = {
                "train": DataLoader(
                    MNIST(
                        os.getcwd(), train=False, download=True, transform=ToTensor()
                    ),
                    batch_size=32,
                ),
                "valid": DataLoader(
                    MNIST(
                        os.getcwd(), train=False, download=True, transform=ToTensor()
                    ),
                    batch_size=32,
                ),
            }

            model = nn.Sequential(Flatten(), nn.Linear(784, 512), nn.ReLU(), nn.Linear(512, 10))
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
            runner = dl.SupervisedRunner()
            runner.train(
                model=model,
                callbacks=[dl.OnnxCallback(input_key="features", logdir="./logs")],
                loaders=loaders,
                criterion=criterion,
                optimizer=optimizer,
                num_epochs=1,
                logdir="./logs",
            )
    """

    def __init__(
        self,
        input_key: str,
        logdir: Union[str, Path] = None,
        filename: str = "onnx.py",
        method_name: str = "forward",
        input_names: Iterable = None,
        output_names: List[str] = None,
        dynamic_axes: Union[Dict[str, int], Dict[str, Dict[str, int]]] = None,
        opset_version: int = 9,
        do_constant_folding: bool = False,
        verbose: bool = False,
    ):
        """Init."""
        super().__init__(order=CallbackOrder.ExternalExtra, node=CallbackNode.Master)
        if logdir is not None:
            self.filename = str(Path(logdir) / filename)
        else:
            self.filename = filename
        # self.input_key = [input_key] if isinstance(input_key, str) else input_key
        self.input_key = input_key
        self.method_name = method_name
        self.input_names = input_names
        self.output_names = output_names
        self.dynamic_axes = dynamic_axes
        self.opset_version = opset_version
        self.do_constant_folding = do_constant_folding
        self.verbose = verbose

    def on_stage_end(self, runner: "IRunner") -> None:
        """
        On stage end action.

        Args:
            runner: runner for experiment
        """
        model = runner.model
        batch = runner.engine.sync_device(runner.batch[self.input_key])
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
            verbose=self.verbose,
        )


__all__ = ["OnnxCallback"]
