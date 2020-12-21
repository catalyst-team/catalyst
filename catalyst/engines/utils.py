from typing import Union
import re

from catalyst.core.engine import IEngine
from catalyst.engines.device import DeviceEngine
from catalyst.engines.distributed import DistributedDataParallelEngine
from catalyst.engines.parallel import DataParallelEngine
from catalyst.settings import IS_CUDA_AVAILABLE, NUM_CUDA_DEVICES


def process_engine(engine: Union[str, IEngine, None]) -> IEngine:
    if isinstance(engine, IEngine):
        return engine

    if not engine:
        # TODO: should be used ddp if have enough GPU (>2) ?
        return DeviceEngine("cuda:0" if IS_CUDA_AVAILABLE else "cpu")

    if engine == "dp" and NUM_CUDA_DEVICES > 2:
        return DataParallelEngine()
    elif engine == "ddp" and NUM_CUDA_DEVICES > 2:
        return DistributedDataParallelEngine()
    elif (
        engine == "cpu"
        # TODO: probably fix pattern str
        or re.match(r"cuda\:\d", str(engine))
    ):
        return DeviceEngine(engine)
    elif engine == "cuda":
        return DeviceEngine("cuda:0")
    else:
        # TODO: should be used ddp if have enough GPU (>2) ?
        return DeviceEngine("cuda:0" if IS_CUDA_AVAILABLE else "cpu")
