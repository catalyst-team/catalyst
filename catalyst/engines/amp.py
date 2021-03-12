import torch
import torch.cuda.amp as amp
from torch.nn.parallel import DataParallel

from catalyst.engines.torch import DeviceEngine, DistributedDataParallelEngine


class AMPEngine(DeviceEngine):
    """@TODO: docs.

    Args:
        device: used device, default is `"cuda"`.
    """

    def __init__(self, device: str = "cuda"):
        """Init."""
        super().__init__(device)
        self.scaler = amp.GradScaler()

    def __repr__(self) -> str:  # noqa: D105
        return f"{self.__class__.__name__}(device='{self.device}')"

    def backward_loss(self, loss, model, optimizer) -> None:
        """@TODO: docs."""
        self.scaler.scale(loss).backward()

    def optimizer_step(self, loss, model, optimizer) -> None:
        """@TODO: docs."""
        self.scaler.step(optimizer)
        self.scaler.update()

    # TODO: should be used with forward method? (similar to criterion)
    def autocast(self):
        """@TODO: docs."""
        return amp.autocast()


class DataParallelAMPEngine(AMPEngine):
    """@TODO: docs."""

    def __init__(self):
        """@TODO: docs."""
        super().__init__(f"cuda:{torch.cuda.current_device()}")
        self.device_count = torch.cuda.device_count()

    def __repr__(self) -> str:  # noqa: D105
        return f"{self.__class__.__name__}(device='{self.device}')"

    def init_components(
        self, model_fn=None, criterion_fn=None, optimizer_fn=None, scheduler_fn=None,
    ):
        """@TODO: docs."""
        model = model_fn()
        model = self.sync_device(model)
        model = DataParallel(model)

        # criterion
        criterion = criterion_fn()
        criterion = self.sync_device(criterion)
        # optimizer
        optimizer = optimizer_fn()
        optimizer = self.sync_device(optimizer)
        # scheduler
        scheduler = scheduler_fn()
        scheduler = self.sync_device(scheduler)

        return model, criterion, optimizer, scheduler


class DistributedDataParallelAMPEngine(DistributedDataParallelEngine):
    """@TODO: docs.

    Args:
        address: process address to use (required for PyTorch backend), default is `"localhost"`.
        port: process port to listen (required for PyTorch backend), default is `"12345"`.
        backend: multiprocessing backend to use, default is `"nccl"`.
        world_size: number of processes.
    """

    def __init__(
        self,
        address: str = "localhost",
        port: str = "12345",
        backend: str = "nccl",
        world_size: int = None,
    ):
        """Init."""
        super().__init__(address, port, backend, world_size)
        self.scaler = amp.GradScaler()

    def __repr__(self):  # noqa: D105
        return (
            f"{self.__class__.__name__}(address={self.address}, "
            f"port={self.port}, backend='{self.backend}',"
            f"rank={self._rank}, world_size={self._world_size})"
        )

    def backward_loss(self, loss, model, optimizer) -> None:
        """@TODO: docs."""
        self.scaler.scale(loss).backward()

    def optimizer_step(self, loss, model, optimizer) -> None:
        """@TODO: docs."""
        self.scaler.step(optimizer)
        self.scaler.update()

    # TODO: should be used with forward method? (similar to criterion)
    def autocast(self):
        """@TODO: docs."""
        return amp.autocast()


__all__ = ["AMPEngine", "DataParallelAMPEngine", "DistributedDataParallelAMPEngine"]
