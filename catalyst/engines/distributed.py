from typing import Any, List
import os
import pickle

import torch
import torch.distributed as dist
import torch.nn as nn

from catalyst.engines.device import DeviceEngine


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
    Source: https://github.com/facebookresearch/detr/blob/master/util/misc.py#L88-L128

    Args:
        data: any picklable object

    Returns:
        list of data gathered from each process.
    """
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
    for _ in size_list:
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


class DistributedDeviceEngine(DeviceEngine):
    def __init__(
        self,
        address: str = "localhost",
        port: str = "12345",
        backend: str = "nccl",
    ):
        self.device = "ddp"
        self.address = address
        self.port = port
        self.backend = backend

    def __repr__(self):
        return (
            f"DistributedDeviceEngine(address={self.address},"
            f"port={self.port},backend='{self.backend}')"
        )

    def setup_experiment(self, rank: int, world_size: int):
        """
        Args:
            rank: process rank
            world_size: total number of processes in experiment
        """
        os.environ["MASTER_ADDR"] = str(self.address)
        os.environ["MASTER_PORT"] = str(self.port)
        dist.init_process_group(self.backend, rank=rank, world_size=world_size)

    def cleanup(self):
        dist.destroy_process_group()

    def sync_metric(self):
        pass

    # TODO: maybe handle unpacking DataParallel to simple nn.Module
    def load_checkpoint(
        self,
        file: str,
        model: nn.DataParallel,
        optimizer: nn.Module = None,
        criterion=None,
        scheduler=None,
    ):
        content = torch.load(file)

        if "model_state_dict" in content:
            model.module.load_state_dict(content["model_state_dict"])

        if "optimizer_state_dict" in content and optimizer is not None:
            optimizer.load_state_dict(content["optimizer_state_dict"])

        if "criterion_state_dict" in content and criterion is not None:
            criterion.load_state_dict(content["criterion_state_dict"])

        if "scheduler_state_dict" in content and scheduler is not None:
            scheduler.load_state_dict(content["scheduler_state_dict"])
