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
        DataParallelAPEXEngine,
        DataParallelApexEngine,
        DistributedDataParallelAPEXEngine,
        DistributedDataParallelApexEngine,
    )

if SETTINGS.fairscale_required:
    from catalyst.engines.fairscale import (
        PipelineParallelFairScaleEngine,
        SharedDataParallelFairScaleEngine,
        SharedDataParallelFairScaleAMPEngine,
        FullySharedDataParallelFairScaleEngine,
    )

if SETTINGS.deepspeed_required:
    from catalyst.engines.deepspeed import DistributedDataParallelDeepSpeedEngine


__all__ = [
    "IEngine",
    "DeviceEngine",
    "DataParallelEngine",
    "DistributedDataParallelEngine",
]

if SETTINGS.amp_required:
    __all__ += ["AMPEngine", "DataParallelAMPEngine", "DistributedDataParallelAMPEngine"]

if SETTINGS.apex_required:
    __all__ += [
        "APEXEngine",
        "DataParallelApexEngine",
        "DataParallelAPEXEngine",
        "DistributedDataParallelApexEngine",
        "DistributedDataParallelAPEXEngine",
    ]

if SETTINGS.fairscale_required:
    __all__ += [
        "PipelineParallelFairScaleEngine",
        "SharedDataParallelFairScaleEngine",
        "SharedDataParallelFairScaleAMPEngine",
        "FullySharedDataParallelFairScaleEngine",
    ]

if SETTINGS.deepspeed_required:
    __all__ += [
        "DistributedDataParallelDeepSpeedEngine",
    ]
