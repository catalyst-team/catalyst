from typing import Any, Dict, Union

import torch
from torch import nn
import torch.cuda.amp as amp

from catalyst.engines.torch import DeviceEngine, DistributedDataParallelEngine
from catalyst.typing import Model, Optimizer


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

        runner = dl.SupervisedRunner()
        runner.train(
            engine=dl.AMPEngine("cuda:1"),
            ...
        )

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
        # TODO: add OptimizerWithScaler abstraction?
        self.scaler_kwargs = scaler_kwargs
        self.scaler = amp.GradScaler(**self.scaler_kwargs)

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"{self.__class__.__name__}(device='{self._device}', "
            f"scaler_kwargs={self.scaler_kwargs})"
        )

    def backward_loss(self, loss: torch.Tensor, model: Model, optimizer: Optimizer) -> None:
        """Abstraction over ``loss.backward()`` step."""
        self.scaler.scale(loss).backward()

    def optimizer_step(self, loss: torch.Tensor, model: Model, optimizer: Optimizer) -> None:
        """Abstraction over ``optimizer.step()`` step."""
        self.scaler.step(optimizer)
        self.scaler.update()

    def pack_checkpoint(
        self, model=None, criterion=None, optimizer=None, scheduler=None, **kwargs
    ) -> Dict:
        """
        Packs ``model``, ``criterion``, ``optimizer``, ``scheduler``
        and some extra info ``**kwargs`` to torch-based checkpoint.

        Args:
            model: torch model
            criterion: torch criterion
            optimizer: torch optimizer
            scheduler: torch scheduler
            **kwargs: some extra info to pack

        Returns:
            torch-based checkpoint with ``model_state_dict``,
            ``criterion_state_dict``, ``optimizer_state_dict``,
            ``scheduler_state_dict`` keys.
        """
        checkpoint = {"scaler": self.scaler.state_dict()}
        checkpoint = super().pack_checkpoint(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            **checkpoint,
        )
        return checkpoint

    def unpack_checkpoint(
        self,
        checkpoint: Dict,
        model=None,
        criterion=None,
        optimizer=None,
        scheduler=None,
        **kwargs,
    ) -> None:
        """Load checkpoint from file and unpack the content to a model
        (if not None), criterion (if not None), optimizer (if not None),
        scheduler (if not None).

        Args:
            checkpoint: checkpoint to load
            model: model where should be updated state
            criterion: criterion where should be updated state
            optimizer: optimizer where should be updated state
            scheduler: scheduler where should be updated state
            kwargs: extra arguments
        """
        super().unpack_checkpoint(
            checkpoint,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            **kwargs,
        )

        if "scaler" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler"])

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

        runner = dl.SupervisedRunner()
        runner.train(
            engine=dl.DataParallelAMPEngine(),
            ...
        )

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
            f"{self.__class__.__name__}(device='{self._device}', "
            f"scaler_kwargs={self.scaler_kwargs})"
        )

    def init_components(
        self, model_fn=None, criterion_fn=None, optimizer_fn=None, scheduler_fn=None
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
        sync_bn: boolean flag for batchnorm synchonization during disributed training.
            if True, applies PyTorch `convert_sync_batchnorm`_ to the model for native torch
            distributed only. Default, False.
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

        runner = dl.SupervisedRunner()
        runner.train(
            engine=dl.DistributedDataParallelAMPEngine(),
            ...
        )

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

    .. _convert_sync_batchnorm:
        https://pytorch.org/docs/stable/generated/torch.nn.SyncBatchNorm.html#
        torch.nn.SyncBatchNorm.convert_sync_batchnorm
    """

    def __init__(
        self,
        address: str = None,
        port: Union[str, int] = None,
        sync_bn: bool = False,
        ddp_kwargs: Dict[str, Any] = None,
        process_group_kwargs: Dict[str, Any] = None,
        scaler_kwargs: Dict[str, Any] = None,
    ):
        """Init."""
        super().__init__(
            address=address,
            port=port,
            sync_bn=sync_bn,
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

    def backward_loss(self, loss: torch.Tensor, model: Model, optimizer: Optimizer) -> None:
        """Abstraction over ``loss.backward()`` step."""
        self.scaler.scale(loss).backward()

    def optimizer_step(self, loss: torch.Tensor, model: Model, optimizer: Optimizer) -> None:
        """Abstraction over ``optimizer.step()`` step."""
        self.scaler.step(optimizer)
        self.scaler.update()

    def pack_checkpoint(
        self, model=None, criterion=None, optimizer=None, scheduler=None, **kwargs
    ) -> Dict:
        """
        Packs ``model``, ``criterion``, ``optimizer``, ``scheduler``
        and some extra info ``**kwargs`` to torch-based checkpoint.

        Args:
            model: torch model
            criterion: torch criterion
            optimizer: torch optimizer
            scheduler: torch scheduler
            **kwargs: some extra info to pack

        Returns:
            torch-based checkpoint with ``model_state_dict``,
            ``criterion_state_dict``, ``optimizer_state_dict``,
            ``scheduler_state_dict`` keys.
        """
        checkpoint = {"scaler": self.scaler.state_dict()}
        checkpoint = super().pack_checkpoint(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            **checkpoint,
        )
        return checkpoint

    def unpack_checkpoint(
        self,
        checkpoint: Dict,
        model=None,
        criterion=None,
        optimizer=None,
        scheduler=None,
        **kwargs,
    ) -> None:
        """Load checkpoint from file and unpack the content to a model
        (if not None), criterion (if not None), optimizer (if not None),
        scheduler (if not None).

        Args:
            checkpoint: checkpoint to load
            model: model where should be updated state
            criterion: criterion where should be updated state
            optimizer: optimizer where should be updated state
            scheduler: scheduler where should be updated state
            kwargs: extra arguments
        """
        super().unpack_checkpoint(
            checkpoint,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            **kwargs,
        )

        if "scaler" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler"])

    def autocast(self):
        """AMP context"""
        return amp.autocast()


__all__ = ["AMPEngine", "DataParallelAMPEngine", "DistributedDataParallelAMPEngine"]
