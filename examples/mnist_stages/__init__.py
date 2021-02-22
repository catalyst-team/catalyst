# flake8: noqa

from catalyst.registry import Registry

from .runner import CustomSupervisedConfigRunner
from .model import SimpleNet

Registry(CustomSupervisedConfigRunner)
Registry(SimpleNet)
