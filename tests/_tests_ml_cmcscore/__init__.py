# flake8: noqa
from catalyst import registry
from catalyst.dl import SupervisedRunner as Runner

from .experiment import Experiment
from .net import Net

registry.Model(Net)
