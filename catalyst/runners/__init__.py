# flake8: noqa


from catalyst.runners.supervised import ISupervisedRunner
from catalyst.runners.self_supervised import ISelfSupervisedRunner
from catalyst.runners.runner import Runner, SelfSupervisedRunner, SupervisedRunner
from catalyst.runners.config import (
    ConfigRunner,
    SupervisedConfigRunner,
    SelfSupervisedConfigRunner,
)

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
        "SelfSupervisedRunner",
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
        "SelfSupervisedRunner",
        "SelfSupervisedConfigRunner",
    ]
