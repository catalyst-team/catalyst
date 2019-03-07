# flake8: noqa
from catalyst.contrib.registry import Registry
from catalyst.dl.experiments.runner import SupervisedRunner as Runner
from .experiment import Experiment
from .model import SimpleNet


Registry.model(SimpleNet)
