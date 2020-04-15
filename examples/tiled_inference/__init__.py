from catalyst.contrib.dl.experiment import (
    TiledInferenceExperiment as Experiment
)
from catalyst.dl.runner import SupervisedRunner as Runner
from catalyst.dl import registry
import segmentation_models_pytorch as smp

registry.Model(smp.FPN)
