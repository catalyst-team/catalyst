# flake8: noqa

from catalyst import registry
from catalyst.dl import SupervisedRunner as Runner

from .experiment import SimpleExperiment
from .model import SimpleNet

registry.REGISTRY.add(SimpleNet)
registry.REGISTRY.add(SimpleExperiment)
