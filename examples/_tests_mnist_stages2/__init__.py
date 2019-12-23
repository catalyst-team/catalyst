# flake8: noqa
from catalyst.contrib.runner import SupervisedAlchemyRunner as Runner
from catalyst.dl import registry  # , SupervisedRunner as Runner
from .experiment import Experiment
from .model import SimpleNet

registry.Model(SimpleNet)
