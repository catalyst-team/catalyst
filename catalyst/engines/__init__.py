# flake8: noqa
import re
import torch
from typing import Union

from catalyst.core.engine import IEngine, Engine

from catalyst.engines.device import DeviceEngine
from catalyst.engines.parallel import DataParallelEngine
from catalyst.engines.distributed import DistributedDataParallelEngine

from catalyst.settings import IS_CUDA_AVAILABLE, NUM_CUDA_DEVICES


def process_engine(engine: Union[str, IEngine, None] = None) -> IEngine:
    """Generate engine from string.

    Args:
        engine (str or IEngine): engine definition.
            If `None` then will be used default available device
            - for two or move available GPUs will be returned
            `DataParallelEngine`, otherwise will be returned
            `DeviceEngine` with GPU or CPU.
            If engine is an instance of `IEngine` then will be
            returned the same object.
            Default is `None`.

    Returns:
        IEngine object
    """
    default_engine = DeviceEngine("cuda" if IS_CUDA_AVAILABLE else "cpu")
    use_engine = None

    if engine is None:
        if NUM_CUDA_DEVICES > 1:
            use_engine = DataParallelEngine()
        else:
            use_engine = default_engine
        return use_engine

    if isinstance(engine, IEngine):
        return engine

    if isinstance(engine, str):
        engine = engine.strip().lower()

    if engine == "dp" and NUM_CUDA_DEVICES > 1:
        use_engine = DataParallelEngine()
    elif engine == "ddp" and NUM_CUDA_DEVICES > 1:
        use_engine = DistributedDataParallelEngine()
    elif (
        engine == "cpu"
        or engine == "cuda"
        or (
            re.match(r"cuda\:\d", engine) and int(engine.split(":")[1]) < torch.cuda.device_count()
        )
    ):
        use_engine = DeviceEngine(engine)
    else:
        use_engine = default_engine

    return use_engine


__all__ = [
    "IEngine",
    "Engine",
    "DeviceEngine",
    "DataParallelEngine",
    "DistributedDataParallelEngine",
    "process_engine",
]
