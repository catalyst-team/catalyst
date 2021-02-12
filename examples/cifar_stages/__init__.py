# flake8: noqa

from catalyst import registry

from .runner import Runner
from .model import SimpleNet

registry.REGISTRY.add(SimpleNet)
