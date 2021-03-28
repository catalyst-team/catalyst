from typing import Dict, Optional, TYPE_CHECKING, Union
from pathlib import Path

import torch

from catalyst.core import Callback, CallbackNode, CallbackOrder
from catalyst.utils import quantize_model

if TYPE_CHECKING:
    from catalyst.core import IRunner


class QuantizationCallback(Callback):
    """
    Callback for model quantiztion.

    Args:
        logdir: path to folder for saving
        filename: filename
        qconfig_spec (Dict, optional): quantization config in PyTorch format.
            Defaults to None.
        dtype (Union[str, Optional[torch.dtype]], optional):
            Type of weights after quantization.
            Defaults to "qint8".

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
                    MNIST(os.getcwd(),
                    train=False,
                    download=True,
                    transform=ToTensor()),
                    batch_size=32,
                ),
                "valid": DataLoader(
                    MNIST(os.getcwd(),
                    train=False,
                    download=True,
                    transform=ToTensor()),
                    batch_size=32,
                ),
            }

            model = nn.Sequential(Flatten(), nn.Linear(784, 512), nn.ReLU(), nn.Linear(512, 10))
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
            runner = dl.SupervisedRunner()
            runner.train(
                model=model,
                callbacks=[dl.QuantizationCallback(logdir="./logs")],
                loaders=loaders,
                criterion=criterion,
                optimizer=optimizer,
                num_epochs=1,
                logdir="./logs",
            )
    """

    def __init__(
        self,
        logdir: Union[str, Path] = None,
        filename: str = "quantized.pth",
        qconfig_spec: Dict = None,
        dtype: Union[str, Optional[torch.dtype]] = "qint8",
    ):
        """Init."""
        super().__init__(
            order=CallbackOrder.ExternalExtra, node=CallbackNode.master
        )  # External Extra for applying
        # after CheckpointCallback; node master for saving.
        self.qconfig_spec = qconfig_spec
        self.dtype = dtype
        if logdir is not None:
            self.filename = Path(logdir) / filename
        else:
            self.filename = filename

    def on_stage_end(self, runner: "IRunner") -> None:
        """Event handler."""
        q_model = quantize_model(
            runner.model.cpu(), qconfig_spec=self.qconfig_spec, dtype=self.dtype
        )
        checkpoint = runner.engine.pack_checkpoint(model=q_model)
        runner.engine.save_checkpoint(checkpoint=checkpoint, path=self.filename)


__all__ = ["QuantizationCallback"]
