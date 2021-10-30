# flake8: noqa

from catalyst.core.engine import IEngine

from catalyst.engines.torch import DeviceEngine, DataParallelEngine, DistributedDataParallelEngine

__all__ = [
    "IEngine",
    "DeviceEngine",
    "DataParallelEngine",
    "DistributedDataParallelEngine",
]

from catalyst.settings import SETTINGS


if SETTINGS.amp_required:
    from catalyst.engines.amp import (
        AMPEngine,
        DataParallelAMPEngine,
        DistributedDataParallelAMPEngine,
    )

    __all__ += ["AMPEngine", "DataParallelAMPEngine", "DistributedDataParallelAMPEngine"]

if SETTINGS.apex_required:
    from catalyst.engines.apex import (
        APEXEngine,
        DataParallelAPEXEngine,
        DataParallelApexEngine,
        DistributedDataParallelAPEXEngine,
        DistributedDataParallelApexEngine,
    )

    __all__ += [
        "APEXEngine",
        "DataParallelApexEngine",
        "DataParallelAPEXEngine",
        "DistributedDataParallelApexEngine",
        "DistributedDataParallelAPEXEngine",
    ]

if SETTINGS.deepspeed_required:
    from catalyst.engines.deepspeed import DistributedDataParallelDeepSpeedEngine

    __all__ += [
        "DistributedDataParallelDeepSpeedEngine",
    ]

if SETTINGS.fairscale_required:
    from catalyst.engines.fairscale import (
        PipelineParallelFairScaleEngine,
        SharedDataParallelFairScaleEngine,
        SharedDataParallelFairScaleAMPEngine,
        FullySharedDataParallelFairScaleEngine,
    )

    __all__ += [
        "PipelineParallelFairScaleEngine",
        "SharedDataParallelFairScaleEngine",
        "SharedDataParallelFairScaleAMPEngine",
        "FullySharedDataParallelFairScaleEngine",
    ]

if SETTINGS.xla_required:
    from catalyst.engines.xla import XLAEngine, DistributedXLAEngine

    __all__ += ["XLAEngine", "DistributedXLAEngine"]
