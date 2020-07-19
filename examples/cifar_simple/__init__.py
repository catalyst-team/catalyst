# flake8: noqa

import experiments as exp

from catalyst import registry
from catalyst.dl import SupervisedRunner as Runner

from .experiment import Experiment
from .model import SimpleNet

registry.EXPERIMENT.add_from_module(exp)
