from typing import List, TYPE_CHECKING, Union
from pathlib import Path

import torch

from catalyst.core import Callback, CallbackNode, CallbackOrder
from catalyst.utils.torch import any2device
from catalyst.utils.tracing import trace_model

if TYPE_CHECKING:
    from catalyst.core import IRunner


class TracingCallback(Callback):
    """
    Callback for model tracing.

    Args:
        input_key: input key from ``runner.batch`` to use for model tracing
        logdir: path to folder for saving
        filename: filename
        method_name: Model's method name that will be used as entrypoint during tracing

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

            model = nn.Sequential(
                Flatten(), nn.Linear(784, 512), nn.ReLU(), nn.Linear(512, 10)
            )
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
            runner = dl.SupervisedRunner()
            runner.train(
                model=model,
                callbacks=[dl.TracingCallback(input_key="features", logdir="./logs")],
                loaders=loaders,
                criterion=criterion,
                optimizer=optimizer,
                num_epochs=1,
                logdir="./logs",
            )
    """

    def __init__(
        self,
        input_key: Union[str, List[str]],
        logdir: Union[str, Path] = None,
        filename: str = "traced_model.pth",
        method_name: str = "forward",
    ):
        """
        Callback for model tracing.

        Args:
            input_key: input key from ``runner.batch`` to use for model tracing
            logdir: path to folder for saving
            filename: filename
            method_name: Model's method name that will be used as entrypoint during tracing

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

                model = nn.Sequential(
                    Flatten(), nn.Linear(784, 512), nn.ReLU(), nn.Linear(512, 10)
                )
                criterion = nn.CrossEntropyLoss()
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
                runner = dl.SupervisedRunner()
                runner.train(
                    model=model,
                    callbacks=[dl.TracingCallback(input_key="features", logdir="./logs")],
                    loaders=loaders,
                    criterion=criterion,
                    optimizer=optimizer,
                    num_epochs=1,
                    logdir="./logs",
                )
        """
        super().__init__(order=CallbackOrder.ExternalExtra, node=CallbackNode.Master)
        if logdir is not None:
            self.filename = str(Path(logdir) / filename)
        else:
            self.filename = filename
        self.method_name = method_name

        self.input_key = [input_key] if isinstance(input_key, str) else input_key

    def on_stage_end(self, runner: "IRunner") -> None:
        """
        On stage end action.

        Args:
            runner: runner for experiment
        """
        model = runner.model
        batch = tuple(runner.batch[key] for key in self.input_key)
        batch = any2device(batch, "cpu")
        traced_model = trace_model(model=model, batch=batch, method_name=self.method_name)
        torch.jit.save(traced_model, self.filename)


__all__ = ["TracingCallback"]
