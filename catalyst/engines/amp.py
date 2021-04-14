from typing import Any, Dict, Union

import torch
from torch import nn
import torch.cuda.amp as amp

from catalyst.engines.torch import DeviceEngine, DistributedDataParallelEngine


class AMPEngine(DeviceEngine):
    """Pytorch.AMP single training device engine.

    Args:
        device: used device, default is `"cuda"`.
        scaler_kwargs: parameters for `torch.cuda.amp.GradScaler`.
            Possible parameters:
            https://pytorch.org/docs/stable/amp.html#torch.cuda.amp.GradScaler

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

    def __init__(self, device: str = "cuda", scaler_kwargs: Dict[str, Any] = None):
        """Init."""
        super().__init__(device)
        if scaler_kwargs is None:
            scaler_kwargs = {}
        self.scaler_kwargs = scaler_kwargs
        self.scaler = amp.GradScaler(**self.scaler_kwargs)

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"{self.__class__.__name__}(device='{self.device}', "
            f"scaler_kwargs={self.scaler_kwargs})"
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


class DataParallelAMPEngine(AMPEngine):
    """AMP multi-gpu training device engine.

    Args:
        scaler_kwargs: parameters for `torch.cuda.amp.GradScaler`.
            Possible parameters:
            https://pytorch.org/docs/stable/amp.html#torch.cuda.amp.GradScaler

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

    def __init__(self, scaler_kwargs: Dict[str, Any] = None):
        """Init."""
        super().__init__(f"cuda:{torch.cuda.current_device()}", scaler_kwargs)
        self.device_count = torch.cuda.device_count()

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"{self.__class__.__name__}(device='{self.device}', "
            f"scaler_kwargs={self.scaler_kwargs})"
        )

    def init_components(
        self, model_fn=None, criterion_fn=None, optimizer_fn=None, scheduler_fn=None,
    ):
        """Inits the runs components."""
        model = model_fn()
        model = self.sync_device(model)

        if isinstance(model, nn.Module):
            model = nn.DataParallel(model)
        elif isinstance(model, dict):
            model = {k: nn.DataParallel(v) for k, v in model.items()}
        else:
            raise ValueError("Model should be ``nn.Module`` or ``dict``")

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
        address: address to use for backend.
        port: port to use for backend.
        ddp_kwargs: parameters for `torch.nn.parallel.DistributedDataParallel`.
            More info here:
            https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel
        process_group_kwargs: parameters for `torch.distributed.init_process_group`.
            More info here:
            https://pytorch.org/docs/stable/distributed.html#torch.distributed.init_process_group
        scaler_kwargs: parameters for `torch.cuda.amp.GradScaler`.
            Possible parameters:
            https://pytorch.org/docs/stable/amp.html#torch.cuda.amp.GradScaler

    Examples:

    .. code-block:: python

        from catalyst import dl

        class MyRunner(dl.IRunner):
            # ...
            def get_engine(self):
                return dl.DistributedDataParallelAMPEngine(
                    address="0.0.0.0",
                    port=23234,
                    ddp_kwargs={"find_unused_parameters": False},
                    process_group_kwargs={"port": 12345},
                    scaler_kwargs={"growth_factor": 1.5}
                )
            # ...

    .. code-block:: yaml

        args:
            logs: ...

        model:
            _target_: ...
            ...

        engine:
            _target_: DistributedDataParallelAMPEngine
            address: 0.0.0.0
            port: 23234
            ddp_kwargs:
                find_unused_parameters: false
            process_group_kwargs:
                port: 12345
            scaler_kwargs:
                growth_factor: 1.5

        stages:
            ...

    """

    def __init__(
        self,
        address: str = None,
        port: Union[str, int] = None,
        ddp_kwargs: Dict[str, Any] = None,
        process_group_kwargs: Dict[str, Any] = None,
        scaler_kwargs: Dict[str, Any] = None,
    ):
        """Init."""
        super().__init__(
            address=address,
            port=port,
            ddp_kwargs=ddp_kwargs,
            process_group_kwargs=process_group_kwargs,
        )
        if scaler_kwargs is None:
            scaler_kwargs = {}
        self.scaler_kwargs = scaler_kwargs
        self.scaler = amp.GradScaler(**self.scaler_kwargs)

    def __repr__(self):  # noqa: D105
        return (
            f"{self.__class__.__name__}(address={self.address}, "
            f"port={self.port}, "
            f"ddp_kwargs={self.ddp_kwargs}, "
            f"process_group_kwargs={self.process_group_kwargs}, "
            f"scaler_kwargs={self.scaler_kwargs})"
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
