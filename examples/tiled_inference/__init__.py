import segmentation_models_pytorch as smp

from catalyst.dl import registry
from catalyst.dl.runner import SupervisedRunner as Runner

from .experiment import TiledInferenceExperiment as Experiment

registry.Model(smp.FPN)
