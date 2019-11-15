# flake8: noqa
from catalyst.dl import SupervisedRunner as Runner
from .experiment import Experiment
from catalyst.dl.registry import MODELS

del MODELS["SimpleNet"]  #  TODO: fix hack for duplicate SimpleNet
from .model import SimpleNet
