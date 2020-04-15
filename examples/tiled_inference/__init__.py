import segmentation_models_pytorch as smp

from catalyst.contrib.dl.experiment import (
    TiledInferenceExperiment as Experiment,
)
from catalyst.dl import registry
from catalyst.dl.runner import SupervisedRunner as Runner

registry.Model(smp.FPN)
