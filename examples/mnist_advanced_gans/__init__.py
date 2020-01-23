# flake8: noqa
from catalyst.dl import registry
from catalyst.dl.runner import GanRunner as Runner  # vanilla GAN
# from runners import WGanRunner as Runner  # WGAN/WGAN-GP
# from runners import CGanRunner as Runner  # vanilla GAN + one-hot class condition
# from runners import ICGanRunner as Runner  # vanilla GAN + same class image condition
from .experiment import MnistGanExperiment as Experiment
# from .experiment import DAGANMnistGanExperiment as Experiment
from . import callbacks, criterion, models
registry.CALLBACKS.add_from_module(callbacks)
registry.CRITERIONS.add_from_module(criterion)
registry.MODELS.add_from_module(models)
