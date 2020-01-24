# flake8: noqa
from catalyst.dl import registry
from catalyst.dl.runner import GanRunner as Runner
from experiment import MnistGanExperiment as Experiment
import transforms
from . import callbacks, models
registry.CALLBACKS.add_from_module(callbacks)
registry.MODELS.add_from_module(models)
