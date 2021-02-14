# flake8: noqa

from catalyst import registry

from .runner import CustomSupervisedConfigRunner
from .model import SimpleNet

registry.REGISTRY.add(CustomSupervisedConfigRunner)
registry.REGISTRY.add(SimpleNet)
