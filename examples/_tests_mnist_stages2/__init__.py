# flake8: noqa
from catalyst.dl import registry
from .experiment import Experiment
from .model import SimpleNet

registry.Model(SimpleNet)
