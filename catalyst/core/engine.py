from typing import Any, Callable, Dict, List, Tuple, Union
from abc import ABC, abstractmethod
from contextlib import contextmanager

import numpy as np
import torch
from torch import nn

from catalyst.typing import Criterion, Device, Model, Optimizer, Scheduler


@contextmanager
def nullcontext(enter_result: Any = None):
    """Context handler."""
    yield enter_result


class IEngine(ABC):
    """
    An abstraction that syncs experiment run with
    different hardware-specific configurations.

    - cpu
    - single-gpu
    - multi-gpu
    - amp (nvidia, torch)
    - ddp (torch, etc)

    Abstraction, please check out implementations for more details:

        - :py:mod:`catalyst.engines.amp.AMPEngine`
        - :py:mod:`catalyst.engines.apex.APEXEngine`
        - :py:mod:`catalyst.engines.torch.DeviceEngine`
    """

    @property
    @abstractmethod
    def device(self) -> Device:
        """Pytorch device."""
        pass

    @property
    @abstractmethod
    def rank(self) -> int:
        """Process rank for distributed training."""
        pass

    @property
    @abstractmethod
    def world_size(self) -> int:
        """Process world size for distributed training."""
        pass

    @property
    def is_ddp(self) -> bool:
        """Boolean flag for distributed run."""
        return self.rank > -1

    @property
    def is_master_process(self) -> bool:
        """Checks if a process is master process.
        Should be implemented only for distributed training (ddp).
        For non distributed training should always return `True`.

        Returns:
            `True` if current process is a master process in other cases return `False`.
        """
        # -1 for non-distributed setup
        # 0 for distributed setup
        return self.rank <= 0

    @property
    def is_worker_process(self) -> bool:
        """Checks if a process is worker process.
        Should be implemented only for distributed training (ddp).
        For non distributed training should always return `False`.

        Returns:
            `True` if current process is a worker process in other cases return `False`.
        """
        return self.rank > 0

    def setup_process(self, rank: int = -1, world_size: int = 1):
        """Initialize DDP variables and processes.

        Args:
            rank: process rank. Default is `-1`.
            world_size: number of devices in netwok to expect for train.
                Default is `1`.
        """
        pass

    def cleanup_process(self):
        """Clean DDP variables and processes."""
        pass

    @abstractmethod
    def sync_device(
        self, tensor_or_module: Union[Dict, List, Tuple, np.ndarray, torch.Tensor, nn.Module]
    ) -> Union[Dict, List, Tuple, torch.Tensor, nn.Module]:
        """Moves ``tensor_or_module`` to Engine's device.

        Args:
            tensor_or_module: tensor to mode
        """
        pass

    @abstractmethod
    def sync_tensor(self, tensor: torch.Tensor, mode: str) -> torch.Tensor:
        """Syncs ``tensor`` over ``world_size`` in distributed mode."""
        pass

    @abstractmethod
    def init_components(
        self,
        model_fn: Callable = None,
        criterion_fn: Callable = None,
        optimizer_fn: Callable = None,
        scheduler_fn: Callable = None,
    ):
        """Inits the runs components."""
        pass

    @abstractmethod
    def deinit_components(self, runner=None):
        """Deinits the runs components. In distributed mode should destroy process group."""
        pass

    @abstractmethod
    def zero_grad(self, loss: torch.Tensor, model: Model, optimizer: Optimizer) -> None:
        """Abstraction over ``model.zero_grad()`` step.
        Should be overloaded in cases when required to set arguments
        for ``model.zero_grad()`` like `set_to_none=True` or
        you need to use custom scheme which replaces/improves
        `.zero_grad()` method.

        Args:
            loss: tensor with loss value.
            model: model module.
            optimizer: model optimizer.
        """
        pass

    @abstractmethod
    def backward_loss(self, loss: torch.Tensor, model: Model, optimizer: Optimizer) -> None:
        """Abstraction over ``loss.backward()`` step.
        Should be overloaded in cases when required loss scaling.
        Examples - APEX and AMP.

        Args:
            loss: tensor with loss value.
            model: model module.
            optimizer: model optimizer.
        """
        pass

    @abstractmethod
    def optimizer_step(self, loss: torch.Tensor, model: Model, optimizer: Optimizer) -> None:
        """Abstraction over ``optimizer.step()`` step.
        Should be overloaded in cases when required gradient scaling.
        Example - AMP.

        Args:
            loss: tensor with loss value.
            model: model module.
            optimizer: model optimizer.
        """
        pass

    @abstractmethod
    def pack_checkpoint(
        self,
        model: Model = None,
        criterion: Criterion = None,
        optimizer: Optimizer = None,
        scheduler: Scheduler = None,
        **kwargs,
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
        """
        pass

    @abstractmethod
    def unpack_checkpoint(
        self,
        checkpoint: Dict,
        model: Model = None,
        criterion: Criterion = None,
        optimizer: Optimizer = None,
        scheduler: Scheduler = None,
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
        pass

    @abstractmethod
    def save_checkpoint(self, checkpoint: Dict, path: str) -> None:
        """Saves checkpoint to a file.

        Args:
            checkpoint: data to save.
            path: filepath where checkpoint should be stored.
        """
        pass

    @abstractmethod
    def load_checkpoint(self, path: str) -> Dict:
        """Load checkpoint from path.

        Args:
            path: checkpoint file to load
        """
        pass

    def autocast(self, *args, **kwargs):
        """AMP scaling context.
        Default autocast context does not scale anything.

        Args:
            *args: some args
            **kwargs: some kwargs

        Returns:
            context
        """
        return nullcontext()


__all__ = ["IEngine"]
