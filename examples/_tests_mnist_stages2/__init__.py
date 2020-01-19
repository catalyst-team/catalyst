# flake8: noqa
from catalyst.dl import registry, SupervisedRunner as Runner
from .experiment import Experiment
from .model import SimpleNet
from .transform import TensorToImage, ToTensor

registry.Model(SimpleNet)
registry.Transform(TensorToImage)
registry.Transform(ToTensor)
