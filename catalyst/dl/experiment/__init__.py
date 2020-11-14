# flake8: noqa

from catalyst.dl.experiment.config import ConfigExperiment
from catalyst.dl.experiment.experiment import Experiment
from catalyst.dl.experiment.supervised import SupervisedExperiment

from catalyst.tools.settings import IS_HYDRA_AVAILABLE

if IS_HYDRA_AVAILABLE:
    from catalyst.dl.experiment.hydra_config import HydraConfigExperiment

    __all__ = [
        "ConfigExperiment",
        "Experiment",
        "SupervisedExperiment",
        "HydraConfigExperiment",
    ]
else:
    __all__ = ["ConfigExperiment", "Experiment", "SupervisedExperiment"]
