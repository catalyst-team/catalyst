# flake8: noqa

import experiments as exp

from catalyst.dl import registry, SupervisedRunner as Runner

from .experiment import Experiment
from .model import SimpleNet

registry.EXPERIMENT.add_from_module(exp)
