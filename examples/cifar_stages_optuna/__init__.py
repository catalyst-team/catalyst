# flake8: noqa

from catalyst import registry

from .experiment import Experiment
from .model import SimpleNet

registry.REGISTRY.add(SimpleNet)
