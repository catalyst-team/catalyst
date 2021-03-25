import torch
import torch.cuda.amp as amp
from torch.nn.parallel import DataParallel

from catalyst.engines.torch import DeviceEngine, DistributedDataParallelEngine


class AMPEngine(DeviceEngine):
    """Pytorch.AMP single training device engine.

    Args:
        device: used device, default is `"cuda"`.

    Examples:

    .. code-block:: python

        from catalyst import dl

        class MyRunner(dl.IRunner):
            # ...
            def get_engine(self):
                return dl.AMPEngine("cuda:1")
            # ...

    .. code-block:: yaml

        args:
            logs: ...

        model:
            _target_: ...
            ...

        engine:
            _target_: AMPEngine
            device: cuda:1

        stages:
            ...

    """

    def __init__(self, device: str = "cuda"):
        """Init."""
        super().__init__(device)
        self.scaler = amp.GradScaler()

    def __repr__(self) -> str:  # noqa: D105
        return f"{self.__class__.__name__}(device='{self.device}')"

    def backward_loss(self, loss, model, optimizer) -> None:
        """Abstraction over ``loss.backward()`` step."""
        self.scaler.scale(loss).backward()

    def optimizer_step(self, loss, model, optimizer) -> None:
        """Abstraction over ``optimizer.step()`` step."""
        self.scaler.step(optimizer)
        self.scaler.update()

    # TODO: should be used with forward method? (similar to criterion)
    def autocast(self):
        """AMP context"""
        return amp.autocast()


class DataParallelAMPEngine(AMPEngine):
    """AMP multi-gpu training device engine.

    Examples:

    .. code-block:: python

        from catalyst import dl

        class MyRunner(dl.IRunner):
            # ...
            def get_engine(self):
                return dl.DataParallelAMPEngine()
            # ...

    .. code-block:: yaml

        args:
            logs: ...

        model:
            _target_: ...
            ...

        engine:
            _target_: DataParallelAMPEngine

        stages:
            ...

    """

    def __init__(self):
        """Init."""
        super().__init__(f"cuda:{torch.cuda.current_device()}")
        self.device_count = torch.cuda.device_count()

    def __repr__(self) -> str:  # noqa: D105
        return f"{self.__class__.__name__}(device='{self.device}')"

    def init_components(
        self, model_fn=None, criterion_fn=None, optimizer_fn=None, scheduler_fn=None,
    ):
        """Inits the runs components."""
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
    """Distributed AMP multi-gpu training device engine.

    Args:
        address (str): process address to use (required for PyTorch backend), default is `"localhost"`.
        port (str or int): process port to listen (required for PyTorch backend), default is `"12345"`.
        backend (str): multiprocessing backend to use, default is `"nccl"`.
        world_size (int): number of processes.

    Examples:

    .. code-block:: python

        from catalyst import dl

        class MyRunner(dl.IRunner):
            # ...
            def get_engine(self):
                return dl.DistributedDataParallelAMPEngine(port=12345)
            # ...

    .. code-block:: yaml

        args:
            logs: ...

        model:
            _target_: ...
            ...

        engine:
            _target_: DistributedDataParallelAMPEngine
            port: 12345

        stages:
            ...

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
        """Abstraction over ``loss.backward()`` step."""
        self.scaler.scale(loss).backward()

    def optimizer_step(self, loss, model, optimizer) -> None:
        """Abstraction over ``optimizer.step()`` step."""
        self.scaler.step(optimizer)
        self.scaler.update()

    # TODO: should be used with forward method? (similar to criterion)
    def autocast(self):
        """AMP context"""
        return amp.autocast()


__all__ = ["AMPEngine", "DataParallelAMPEngine", "DistributedDataParallelAMPEngine"]
