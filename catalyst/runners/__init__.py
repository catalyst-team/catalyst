# flake8: noqa


from catalyst.runners.supervised import ISupervisedRunner
from catalyst.runners.contrastive import IContrastiveRunner
from catalyst.runners.runner import Runner, ContrastiveRunner, SupervisedRunner
from catalyst.runners.config import ConfigRunner, SupervisedConfigRunner, ContrastiveConfigRunner

from catalyst.settings import SETTINGS


if SETTINGS.hydra_required:
    from catalyst.runners.hydra import HydraRunner, SupervisedHydraRunner

    __all__ = [
        "Runner",
        "ISupervisedRunner",
        "IContrastiveRunner",
        "SupervisedRunner",
        "ConfigRunner",
        "SupervisedConfigRunner",
        "HydraRunner",
        "SupervisedHydraRunner",
        "ContrastiveRunner",
        "ContrastiveConfigRunner",
    ]
else:
    __all__ = [
        "Runner",
        "ISupervisedRunner",
        "IContrastiveRunner",
        "SupervisedRunner",
        "ConfigRunner",
        "SupervisedConfigRunner",
        "ContrastiveRunner",
        "ContrastiveConfigRunner",
    ]
