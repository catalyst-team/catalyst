# flake8: noqa


from catalyst.runners.supervised import ISupervisedRunner
from catalyst.runners.contrastive import ISelfSupervisedRunner
from catalyst.runners.runner import Runner, ContrastiveRunner, SupervisedRunner
from catalyst.runners.config import ConfigRunner, SupervisedConfigRunner, SelfSupervisedConfigRunner

from catalyst.settings import SETTINGS


if SETTINGS.hydra_required:
    from catalyst.runners.hydra import HydraRunner, SupervisedHydraRunner

    __all__ = [
        "Runner",
        "ISupervisedRunner",
        "ISelfSupervisedRunner",
        "SupervisedRunner",
        "ConfigRunner",
        "SupervisedConfigRunner",
        "HydraRunner",
        "SupervisedHydraRunner",
        "ContrastiveRunner",
        "SelfSupervisedConfigRunner",
    ]
else:
    __all__ = [
        "Runner",
        "ISupervisedRunner",
        "ISelfSupervisedRunner",
        "SupervisedRunner",
        "ConfigRunner",
        "SupervisedConfigRunner",
        "ContrastiveRunner",
        "SelfSupervisedConfigRunner",
    ]
