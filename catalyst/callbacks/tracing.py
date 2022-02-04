# flake8: noqa
# from typing import List, TYPE_CHECKING, Union
# from pathlib import Path

# import torch

# from catalyst.core import Callback, CallbackOrder
# from catalyst.utils.tracing import trace_model

# if TYPE_CHECKING:
#     from catalyst.core import IRunner


# class TracingCallback(Callback):
#     """
#     Callback for model tracing.

#     Args:
#         input_key: input key from ``runner.batch`` to use for model tracing
#         logdir: path to folder for saving
#         filename: filename
#         method_name: Model's method name that will be used as entrypoint during tracing

#     Example:

#         .. code-block:: python

#             import os

#             import torch
#             from torch import nn
#             from torch.utils.data import DataLoader

#             from catalyst import dl
#             from catalyst.data import ToTensor
#             from catalyst.contrib.datasets import MNIST
#             from catalyst.contrib.layers import Flatten

#             loaders = {
#                 "train": DataLoader(
#                     MNIST(
#                         os.getcwd(), train=False
#                     ),
#                     batch_size=32,
#                 ),
#                 "valid": DataLoader(
#                     MNIST(
#                         os.getcwd(), train=False
#                     ),
#                     batch_size=32,
#                 ),
#             }

#             model = nn.Sequential(
#                 Flatten(), nn.Linear(784, 512), nn.ReLU(), nn.Linear(512, 10)
#             )
#             criterion = nn.CrossEntropyLoss()
#             optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
#             runner = dl.SupervisedRunner()
#             runner.train(
#                 model=model,
#                 callbacks=[dl.TracingCallback(input_key="features", logdir="./logs")],
#                 loaders=loaders,
#                 criterion=criterion,
#                 optimizer=optimizer,
#                 num_epochs=1,
#                 logdir="./logs",
#             )
#     """

#     def __init__(
#         self,
#         input_key: Union[str, List[str]],
#         logdir: Union[str, Path] = None,
#         filename: str = "traced_model.pth",
#         method_name: str = "forward",
#     ):
#         """
#         Callback for model tracing.

#         Args:
#             input_key: input key from ``runner.batch`` to use for model tracing
#             logdir: path to folder for saving
#             filename: filename
#             method_name: Model's method name
#                 that will be used as entrypoint during tracing

#         Example:
#             .. code-block:: python

#                 import os

#                 import torch
#                 from torch import nn
#                 from torch.utils.data import DataLoader

#                 from catalyst import dl
#                 from catalyst.data import ToTensor
#                 from catalyst.contrib.datasets import MNIST
#                 from catalyst.contrib.layers import Flatten

#                 loaders = {
#                     "train": DataLoader(
#                         MNIST(
#                             os.getcwd(), train=False
#                         ),
#                         batch_size=32,
#                     ),
#                     "valid": DataLoader(
#                         MNIST(
#                             os.getcwd(), train=False
#                         ),
#                         batch_size=32,
#                     ),
#                 }

#                 model = nn.Sequential(
#                     Flatten(), nn.Linear(784, 512), nn.ReLU(), nn.Linear(512, 10)
#                 )
#                 criterion = nn.CrossEntropyLoss()
#                 optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
#                 runner = dl.SupervisedRunner()
#                 runner.train(
#                     model=model,
#                     callbacks=[
# dl.TracingCallback(input_key="features", logdir="./logs")],
#                     loaders=loaders,
#                     criterion=criterion,
#                     optimizer=optimizer,
#                     num_epochs=1,
#                     logdir="./logs",
#                 )
#         """
#         super().__init__(order=CallbackOrder.External)
#         if logdir is not None:
#             self.filename = str(Path(logdir) / filename)
#         else:
#             self.filename = filename
#         self.method_name = method_name

#         self.input_key = [input_key] if isinstance(input_key, str) else input_key

#     def on_experiment_end(self, runner: "IRunner") -> None:
#         """Event handler."""
#         # model = runner.engine.sync_device(runner.model)
#         model = runner.model
#         batch = tuple(runner.batch[key] for key in self.input_key)
#         # batch = runner.engine.sync_device(batch)
#         traced_model = trace_model(
#             model=model, batch=batch, method_name=self.method_name
#         )
#         torch.jit.save(traced_model, self.filename)


# __all__ = ["TracingCallback"]
