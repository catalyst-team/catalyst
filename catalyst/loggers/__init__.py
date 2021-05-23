# flake8: noqa

from catalyst.loggers.console import ConsoleLogger
from catalyst.loggers.csv import CSVLogger
from catalyst.loggers.tensorboard import TensorboardLogger

from catalyst.settings import SETTINGS

if SETTINGS.mlflow_required:
    from catalyst.loggers.mlflow import MLflowLogger

if SETTINGS.wandb_required:
    from catalyst.loggers.wandb import WandbLogger

if SETTINGS.neptune_required:
    from catalyst.loggers.neptune import NeptuneLogger

__all__ = ["ConsoleLogger", "CSVLogger", "TensorboardLogger"]


if SETTINGS.mlflow_required:
    __all__ += ["MLflowLogger"]

if SETTINGS.wandb_required:
    __all__ += ["WandbLogger"]

if SETTINGS.neptune_required:
    __all__ += ["NeptuneLogger"]
