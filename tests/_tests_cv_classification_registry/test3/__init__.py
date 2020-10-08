# flake8: noqa

from catalyst import registry

from .experiment import SimpleExperiment
from .model import SimpleNet

registry.Model(SimpleNet)
registry.Experiment(SimpleExperiment)
