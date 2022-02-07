# flake8: noqa

from catalyst.core.engine import Engine

from catalyst.engines.torch import (
    CPUEngine,
    GPUEngine,
    Engine,
    DataParallelEngine,
    DistributedDataParallelEngine,
)

__all__ = [
    "Engine",
    "CPUEngine",
    "GPUEngine",
    "DataParallelEngine",
    "DistributedDataParallelEngine",
]

from catalyst.settings import SETTINGS

if SETTINGS.xla_required:
    from catalyst.engines.torch import DistributedXLAEngine

    __all__ += ["DistributedXLAEngine"]
