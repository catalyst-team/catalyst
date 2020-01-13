# flake8: noqa
from catalyst.dl import registry, SupervisedDLRunner as Runner
from .experiment import Experiment
from .model import SimpleNet

registry.Model(SimpleNet)
