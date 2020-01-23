# flake8: noqa
# TODO: remove this temporary workaround when BatchTransforms will be removed
import sys
sys.path.insert(0, '..')
from mnist_advanced_gans import *

from catalyst.dl.runner import GanRunner as Runner
# from .callbacks import VisualizationCallback
# from .experiment import MnistGanExperiment as Experiment
# from .model import SimpleDiscriminator, SimpleGenerator
