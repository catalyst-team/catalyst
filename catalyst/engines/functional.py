# flake8: noqa
from typing import Any, List, Union
import pickle
import re
import socket

import torch
import torch.distributed as dist

# taken from https://github.com/catalyst-team/catalyst/blob/master/catalyst/utils/distributed.py#L157
# def get_distributed_mean(value: Union[float, torch.Tensor]):
#     """Computes distributed mean among all nodes."""
#     if check_torch_distributed_initialized():
#         # Fix for runtime warning:
#         # To copy construct from a tensor, it is recommended to use
#         # sourceTensor.clone().detach() or
#         # sourceTensor.clone().detach().requires_grad_(True),
#         # rather than torch.tensor(sourceTensor).
#         if torch.is_tensor(value):
#             value = value.clone().detach().to(device=f"cuda:{torch.cuda.current_device()}")
#         else:
#             value = torch.tensor(
#                 value,
#                 dtype=torch.float,
#                 device=f"cuda:{torch.cuda.current_device()}",
#                 requires_grad=False,
#             )
#         torch.distributed.all_reduce(value)
#         value = float(value.item() / torch.distributed.get_world_size())
#     return value

# TODO: add tests for this method
def get_available_port() -> str:
    """Find any free available port to use for training.

    Returns:
        string with available port.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()

    return port


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
        tensor_list.append(
            torch.empty((max_size,), dtype=torch.uint8, device="cuda")
        )

    if local_size != max_size:
        padding = torch.empty(
            size=(max_size - local_size,), dtype=torch.uint8, device="cuda"
        )
        tensor = torch.cat((tensor, padding), dim=0)
    dist.all_gather(tensor_list, tensor)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list
