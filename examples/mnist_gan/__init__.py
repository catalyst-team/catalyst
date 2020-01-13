# flake8: noqa
from catalyst.dl.runner import GanDLRunner as Runner
from .callbacks import VisualizationCallback
from .experiment import MnistGanExperiment as Experiment
from .model import SimpleDiscriminator, SimpleGenerator
