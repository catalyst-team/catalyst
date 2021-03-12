# flake8: noqa

from catalyst.loggers.console import ConsoleLogger
from catalyst.loggers.csv import CSVLogger
from catalyst.loggers.tensorboard import TensorboardLogger

from catalyst.settings import SETTINGS

if SETTINGS.mlflow_required:
    from catalyst.loggers.mlflow import MLflowLogger
