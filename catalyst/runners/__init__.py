# flake8: noqa
import logging

logger = logging.getLogger(__name__)

from catalyst.runners.runner import Runner
from catalyst.runners.supervised import ISupervisedRunner, SupervisedRunner
from catalyst.runners.config import ConfigRunner, SupervisedConfigRunner

from catalyst.settings import IS_HYDRA_AVAILABLE, SETTINGS


if IS_HYDRA_AVAILABLE or SETTINGS.hydra_required:
    try:
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
    except ModuleNotFoundError as ex:
        if SETTINGS.hydra_required:
            logger.warning(
                "catalyst[hydra] requirements are not available, to install them,"
                " run `pip install catalyst[hydra]`."
            )
            raise ex
    except ImportError as ex:
        if SETTINGS.hydra_required:
            logger.warning(
                "catalyst[hydra] requirements are not available, to install them,"
                " run `pip install catalyst[hydra]`."
            )
            raise ex
else:
    __all__ = [
        "Runner",
        "ISupervisedRunner",
        "SupervisedRunner",
        "ConfigRunner",
        "SupervisedConfigRunner",
    ]
