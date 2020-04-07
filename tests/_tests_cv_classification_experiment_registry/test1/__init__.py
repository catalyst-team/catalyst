# flake8: noqa

from catalyst.dl import registry, SupervisedRunner as Runner

from .experiment import SimpleExperiment
from .model import SimpleNet

registry.Model(SimpleNet)
registry.Experiment(SimpleExperiment)
