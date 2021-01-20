from typing import Any, List, Union
import pickle
import re

import torch
import torch.distributed as dist

from catalyst.core.engine import IEngine
from catalyst.engines.device import DeviceEngine
from catalyst.engines.distributed import DistributedDataParallelEngine
from catalyst.engines.parallel import DataParallelEngine
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


def sum_reduce(tensor: torch.Tensor) -> torch.Tensor:
    """Reduce tensor to all processes and compute total (sum) value.

    Args:
        tensor: tensor to reduce.

    Returns:
        reduced tensor
    """
    cloned = tensor.clone()
    dist.all_reduce(cloned, dist.ReduceOp.SUM)
    return cloned


def mean_reduce(tensor: torch.Tensor, world_size: int) -> torch.Tensor:
    """Reduce tensor to all processes and compute mean value.

    Args:
        tensor: tensor to reduce.
        world_size: number of processes in DDP setup.

    Returns:
        reduced tensor
    """
    # TODO: fix division operator for int/long tensors
    reduced = sum_reduce(tensor) / world_size
    return reduced


def all_gather(data: Any) -> List[Any]:
    """Run all_gather on arbitrary picklable data (not necessarily tensors).

    NOTE: if data on different devices then data in resulted list will
        be on the same devices.

    Source: 
        https://github.com/facebookresearch/detr/blob/master/util/misc.py#L88-L128

    Args:
        data: any picklable object

    Returns:
        list of data gathered from each process.
    """  # noqa: W501,W505
    if not dist.is_available() or not dist.is_initialized():
        world_size = 1
    else:
        world_size = dist.get_world_size()

    if world_size == 1:
        return [data]

    # serialized to a Tensor
    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to("cuda")

    # obtain Tensor size of each rank
    local_size = torch.tensor([tensor.numel()], device="cuda")
    size_list = [torch.tensor([0], device="cuda") for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # receiving Tensor from all ranks
    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    tensor_list = []
    for _ in size_list:  # noqa: WPS122
        tensor_list.append(torch.empty((max_size,), dtype=torch.uint8, device="cuda"))

    if local_size != max_size:
        padding = torch.empty(size=(max_size - local_size,), dtype=torch.uint8, device="cuda")
        tensor = torch.cat((tensor, padding), dim=0)
    dist.all_gather(tensor_list, tensor)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list
