# flake8: noqa
from catalyst.dl import registry, SupervisedRunner as Runner

from .experiment import Experiment
from .net import Net

registry.Model(Net)
