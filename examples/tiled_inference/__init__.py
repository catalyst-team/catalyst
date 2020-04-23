import segmentation_models_pytorch as smp

from .experiment import TiledInferenceExperiment as Experiment
from catalyst.dl import registry
from catalyst.dl.runner import SupervisedRunner as Runner

registry.Model(smp.FPN)
