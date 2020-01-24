# flake8: noqa

from albumentations.pytorch import ToTensorV2

from catalyst.dl import registry, SupervisedDLRunner as Runner
from .experiment import Experiment
from .model import SimpleNet

registry.Model(SimpleNet)
registry.Transform(ToTensorV2)
