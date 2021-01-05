# flake8: noqa

import experiments as exp

from catalyst.registry import REGISTRY
from catalyst.dl import SupervisedRunner as Runner

from .experiment import Experiment
from .model import SimpleNet

REGISTRY.add_from_module(exp)
