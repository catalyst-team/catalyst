# flake8: noqa
import re
import torch
from typing import Union

from catalyst.core.engine import IEngine

from catalyst.engines.device import DeviceEngine
from catalyst.engines.parallel import DataParallelEngine
from catalyst.engines.distributed import DistributedDataParallelEngine

from catalyst.settings import (
    IS_CUDA_AVAILABLE,
    NUM_CUDA_DEVICES,
    IS_AMP_AVAILABLE,
    IS_APEX_AVAILABLE,
)

if IS_AMP_AVAILABLE:
    from catalyst.engines.amp import AMPEngine, DistributedDataParallelAMPEngine
else:
    # replacement
    from catalyst.engines.device import DeviceEngine as AMPEngine
    from catalyst.engines.distributed import DistributedDataParallelEngine as DistributedDataParallelAMPEngine

if IS_APEX_AVAILABLE:
    from catalyst.engines.apex import APEXEngine, DistributedDataParallelApexEngine
else:
    # replacement
    from catalyst.engines.device import DeviceEngine as APEXEngine
    from catalyst.engines.distributed import DistributedDataParallelEngine as DistributedDataParallelApexEngine


# TODO: add option to create other engines (amp/apex) from string
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

    if engine.startswith("amp") and IS_AMP_AVAILABLE and IS_CUDA_AVAILABLE:
        # Usage:
        #   amp-cuda:N  - amp with specified CUDA device
        #   amp-ddp     - DDP with amp
        _engine_parts = engine.split("-")
        if _engine_parts[1] == "ddp":
            use_engine = DistributedDataParallelAMPEngine()
        else:
            use_engine = AMPEngine(_engine_parts[1])
    elif engine.startswith("apex") and IS_APEX_AVAILABLE and IS_CUDA_AVAILABLE:
        # Usage:
        #   apex-cuda:N     - apex with default opt level (O1) and specified CUDA device
        #   apex-oN-cuda:M  - apex with specified opt level and specified CUDA device
        #   apex-ddp        - DDP with apex with default opt level (O1)
        #   apex-oN-ddp     - DPP with apex with specified opt level
        _engine_parts = engine.split("-")
        # parse string
        if len(_engine_parts) >= 3 and _engine_parts[1] in {"o0", "o1", "o2", "o3"}:
            _opt_level = _engine_parts[1]
            _device = _engine_parts[2]
        else:
            _opt_level = "o1"
            _device = _engine_parts[1]
        # select engine
        if _device == "ddp":
            use_engine = DistributedDataParallelApexEngine(opt_level=_opt_level)
        else:
            use_engine = APEXEngine(_device, _opt_level.upper())
    elif engine == "dp" and NUM_CUDA_DEVICES > 1:
        use_engine = DataParallelEngine()
    elif engine == "ddp" and NUM_CUDA_DEVICES > 1:
        use_engine = DistributedDataParallelEngine()
    elif (
        engine == "cpu"
        or engine == "cuda"
        or (re.match(r"cuda\:\d", engine) and int(engine.split(":")[1]) < torch.cuda.device_count())
    ):
        use_engine = DeviceEngine(engine)
    else:
        use_engine = default_engine

    return use_engine


__all__ = [
    "IEngine",
    "DeviceEngine",
    "DataParallelEngine",
    "DistributedDataParallelEngine",
    "process_engine",
]
