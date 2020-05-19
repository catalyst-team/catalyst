import segmentation_models_pytorch as smp

from catalyst.dl import registry
from catalyst.dl.runner import SupervisedRunner as Runner  # noqa: F401

from .experiment import TiledInferenceExperiment as Experiment  # noqa: F401

registry.Model(smp.FPN)
