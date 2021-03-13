from typing import Any, Dict
from abc import ABC, abstractmethod
from contextlib import contextmanager

from catalyst.typing import Criterion, Model, Optimizer, Scheduler


@contextmanager
def nullcontext(enter_result=None):
    """@TODO: docs"""
    yield enter_result


# @TODO: should IEngine be ICallback-based?
class IEngine(ABC):
    """
    An abstraction that syncs experiment run with
    different hardware-specific configurations.

    - cpu
    - single-gpu
    - multi-gpu
    - amp (nvidia, torch)
    - ddp (torch, etc)
    """

    # @property
    # @abstractmethod
    # def device(self) -> Device:
    #     pass

    @property
    @abstractmethod
    def rank(self) -> int:
        """"Process rank"""
        pass

    @property
    @abstractmethod
    def world_size(self) -> int:
        """Process world size"""
        # only for ddp
        pass

    @property
    def is_ddp(self) -> bool:
        """Boolean flag for distributed run"""
        return self.rank > -1

    @property
    def is_master_process(self) -> bool:
        """Checks if a process is master process.
        Should be implemented only for DDP setup in other cases should always return True.

        Returns:
            `True` if current process is a master process, otherwise `False`.
        """
        return True

    @property
    def is_worker_process(self) -> bool:
        """Checks if a process is worker process.
        Should be implemented only for DDP setup in other cases should always return False.

        Returns:
            `True` if current process is a worker process, otherwise `False`.
        """
        return False

    @abstractmethod
    def sync_device(self, tensor_or_module: Any) -> Any:
        """@TODO: docs"""
        pass
        # return any2device(batch, self.device)

    @abstractmethod
    def sync_tensor(self, tensor: Any, mode: str) -> Any:
        """@TODO: docs"""
        pass

    @abstractmethod
    def init_components(
        self, model_fn=None, criterion_fn=None, optimizer_fn=None, scheduler_fn=None,
    ):
        """@TODO: docs"""
        pass

    @abstractmethod
    def deinit_components(self):
        """@TODO: docs"""
        # only for ddp
        pass

    @abstractmethod
    def zero_grad(self, loss, model, optimizer) -> None:
        """@TODO: docs"""
        pass

    @abstractmethod
    def backward_loss(self, loss, model, optimizer) -> None:
        """@TODO: docs"""
        pass

    @abstractmethod
    def optimizer_step(self, loss, model, optimizer) -> None:
        """@TODO: docs"""
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
        """@TODO: docs"""
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
        """@TODO: docs"""
        pass

    @abstractmethod
    def save_checkpoint(self, checkpoint: Dict, path: str) -> None:
        """@TODO: docs"""
        pass

    @abstractmethod
    def load_checkpoint(self, path: str) -> Dict:
        """@TODO: docs"""
        pass

    def autocast(self, *args, **kwargs):
        """AMP scaling context. Default autocast context does not scale anything.

        Args:
            *args: some args
            **kwargs: some kwargs

        Returns:
            context
        """
        return nullcontext()
