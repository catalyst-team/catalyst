from catalyst.contrib.registry import Registry

from .experiment import Experiment
from catalyst.dl.experiments.runner import SupervisedRunner as Runner
from .callbacks import VerboseCallback
from .model import SimpleNet

Registry.callback(VerboseCallback)
Registry.model(SimpleNet)

