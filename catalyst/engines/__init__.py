# flake8: noqa

from catalyst.core.engine import IEngine

from catalyst.engines.torch import (
    CPUEngine,
    GPUEngine,
    DeviceEngine,
    DataParallelEngine,
    DistributedDataParallelEngine,
)

__all__ = [
    "IEngine",
    "CPUEngine",
    "GPUEngine",
    "DeviceEngine",
    "DataParallelEngine",
    "DistributedDataParallelEngine",
]

from catalyst.settings import SETTINGS

if SETTINGS.xla_required:
    from catalyst.engines.torch import DistributedXLAEngine

    __all__ += ["DistributedXLAEngine"]
