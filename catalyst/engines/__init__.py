# flake8: noqa

from catalyst.core.engine import IEngine

from catalyst.engines.torch import DeviceEngine, DataParallelEngine, DistributedDataParallelEngine

from catalyst.settings import IS_CUDA_AVAILABLE, NUM_CUDA_DEVICES, SETTINGS


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


def get_engine(fp16: bool = False, apex: bool = False, ddp: bool = False):
    """Default engine based on given arguments.

    Args:
        fp16 (bool): option to use fp16 for training.
            Default is `False`.
        apex (bool): option to use APEX for training.
            Default is `False`.
        ddp (bool): option to use DDP for training.
            Default is `False`.

    Returns:
        IEngine which match requirements.
    """
    has_multiple_gpus = NUM_CUDA_DEVICES > 1
    if not IS_CUDA_AVAILABLE:
        return DeviceEngine("cpu")
    else:
        if fp16 and SETTINGS.amp_required and ddp and has_multiple_gpus:
            return DistributedDataParallelAMPEngine()
        elif apex and SETTINGS.apex_required and ddp and has_multiple_gpus:
            return DistributedDataParallelApexEngine()
        elif fp16 and has_multiple_gpus:
            return DataParallelAMPEngine()
        elif fp16 and NUM_CUDA_DEVICES == 1:
            return AMPEngine()
        elif apex and has_multiple_gpus:
            return DataParallelApexEngine()
        elif apex and NUM_CUDA_DEVICES == 1:
            return APEXEngine()
        elif ddp and has_multiple_gpus:
            return DistributedDataParallelEngine()
        elif has_multiple_gpus:
            return DataParallelEngine()
    return DeviceEngine("cuda")


__all__ = [
    "get_engine",
    "IEngine",
    "DeviceEngine",
    "DataParallelEngine",
    "DistributedDataParallelEngine",
]

if SETTINGS.amp_required:
    __all__ += ["AMPEngine", "DataParallelAMPEngine", "DistributedDataParallelAMPEngine"]

if SETTINGS.apex_required:
    __all__ += ["APEXEngine", "DataParallelApexEngine", "DistributedDataParallelApexEngine"]
