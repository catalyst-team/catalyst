# flake8: noqa
# from catalyst.dl.runner import GanRunner as Runner
# from runners import BaseGANRunner as Runner  # vanilla GAN
# from runners import WGANRunner as Runner  # vanilla WGAN
from runners import WGAN_GP_Runner as Runner  # WGAN-GP
from .callbacks import VisualizationCallback, LipzOptimizerCallback, CriterionWithDiscriminatorCallback
from .experiment import MnistGanExperiment as Experiment
from .model import SimpleDiscriminator, SimpleGenerator
from .criterion import *
