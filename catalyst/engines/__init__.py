# flake8: noqa

from catalyst.core.engine import IEngine

from catalyst.engines.torch import DeviceEngine, DataParallelEngine, DistributedDataParallelEngine

from catalyst.settings import SETTINGS


if SETTINGS.amp_required:
    from catalyst.engines.amp import (
        AMPEngine,
        DataParallelAMPEngine,
        DistributedDataParallelAMPEngine,
    )

if SETTINGS.apex_required:
    from catalyst.engines.apex import (
        APEXEngine,
        DataParallelApexEngine,
        DistributedDataParallelApexEngine,
    )


__all__ = [
    "IEngine",
    "DeviceEngine",
    "DataParallelEngine",
    "DistributedDataParallelEngine",
]

if SETTINGS.amp_required:
    __all__ += ["AMPEngine", "DataParallelAMPEngine", "DistributedDataParallelAMPEngine"]

if SETTINGS.apex_required:
    __all__ += ["APEXEngine", "DataParallelApexEngine", "DistributedDataParallelApexEngine"]
