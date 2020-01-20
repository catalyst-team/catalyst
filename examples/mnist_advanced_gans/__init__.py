# flake8: noqa
# from catalyst.dl.runner import GanRunner as Runner
from runners import GANRunner as Runner  # vanilla GAN
# from runners import WGANRunner as Runner  # WGAN/WGAN-GP
# from runners import CGanRunner as Runner  # vanilla GAN + one-hot class condition
# from runners import ICGanRunner as Runner  # vanilla GAN + same class image condition
from .callbacks import *
# from .experiment import MnistGanExperiment as Experiment
from .experiment import DAGANMnistGanExperiment as Experiment
from .model import SimpleDiscriminator, SimpleGenerator, SimpleCDiscriminator, SimpleCGenerator
from .criterion import *
