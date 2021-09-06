# flake8: noqa


from catalyst.runners.supervised import ISupervisedRunner
from catalyst.runners.contrastive import IContrastiveRunner
from catalyst.runners.runner import Runner, ContrastiveRunner, SupervisedRunner
from catalyst.runners.config import ConfigRunner, SupervisedConfigRunner

from catalyst.settings import SETTINGS


if SETTINGS.hydra_required:
    from catalyst.runners.hydra import HydraRunner, SupervisedHydraRunner

    __all__ = [
        "Runner",
        "ISupervisedRunner",
        "SupervisedRunner",
        "ContrastiveRunner",
        "ConfigRunner",
        "SupervisedConfigRunner",
        "HydraRunner",
        "SupervisedHydraRunner",
        "ContrastiveRunner",
    ]
else:
    __all__ = [
        "Runner",
        "ISupervisedRunner",
        "SupervisedRunner",
        "ConfigRunner",
        "SupervisedConfigRunner",
        "ContrastiveRunner",
    ]
