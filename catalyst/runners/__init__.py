# flake8: noqa

from catalyst.runners.runner import Runner
from catalyst.runners.supervised import ISupervisedRunner, SupervisedRunner
from catalyst.runners.config import ConfigRunner, SupervisedConfigRunner

from catalyst.settings import IS_HYDRA_AVAILABLE


if IS_HYDRA_AVAILABLE:
    from catalyst.runners.hydra import HydraRunner, SupervisedHydraRunner

    __all__ = [
        "Runner",
        "ISupervisedRunner",
        "SupervisedRunner",
        "ConfigRunner",
        "SupervisedConfigRunner",
        "HydraRunner",
        "SupervisedHydraRunner",
    ]
else:
    __all__ = [
        "Runner",
        "ISupervisedRunner",
        "SupervisedRunner",
        "ConfigRunner",
        "SupervisedConfigRunner",
    ]
