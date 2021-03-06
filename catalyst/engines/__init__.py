# flake8: noqa
import re
import torch
from typing import Union

from catalyst.core.engine import IEngine

from catalyst.engines.device import DeviceEngine
from catalyst.engines.parallel import DataParallelEngine
from catalyst.engines.distributed import DistributedDataParallelEngine

from catalyst.settings import IS_CUDA_AVAILABLE, NUM_CUDA_DEVICES, SETTINGS


if SETTINGS.amp_required:
    from catalyst.engines.amp import AMPEngine, DistributedDataParallelAMPEngine

if SETTINGS.apex_required:
    from catalyst.engines.apex import APEXEngine, DistributedDataParallelApexEngine


__all__ = [
    "IEngine",
    "DeviceEngine",
    "DataParallelEngine",
    "DistributedDataParallelEngine",
]

if SETTINGS.amp_required:
    __all__ += ["AMPEngine", "DistributedDataParallelAMPEngine"]

if SETTINGS.apex_required:
    __all__ += ["APEXEngine", "DistributedDataParallelApexEngine"]
