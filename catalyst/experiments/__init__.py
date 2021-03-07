# flake8: noqa

from catalyst.experiments.experiment import Experiment
from catalyst.experiments.auto import AutoCallbackExperiment
from catalyst.experiments.config import ConfigExperiment

from catalyst.settings import IS_HYDRA_AVAILABLE

if IS_HYDRA_AVAILABLE:
    from catalyst.experiments.hydra_config import HydraConfigExperiment

    __all__ = [
        "ConfigExperiment",
        "Experiment",
        "AutoCallbackExperiment",
        "HydraConfigExperiment",
    ]
else:
    __all__ = ["ConfigExperiment", "Experiment", "AutoCallbackExperiment"]
