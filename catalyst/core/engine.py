from typing import Any, Dict
from abc import ABC, abstractmethod
from contextlib import contextmanager


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
        """@TODO: docs"""
        pass

    @property
    @abstractmethod
    def world_size(self) -> int:
        """@TODO: docs"""
        # only for ddp
        pass

    @abstractmethod
    def sync_device(self, tensor_or_module: Any) -> Any:
        """@TODO: docs"""
        pass
        # return any2device(batch, self.device)

    @abstractmethod
    def sync_tensor(self, tensor: Any) -> Any:
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
        self, model=None, criterion=None, optimizer=None, scheduler=None, **kwargs,
    ) -> Dict:
        """@TODO: docs"""
        pass

    @abstractmethod
    def unpack_checkpoint(
        self,
        checkpoint: Dict,
        model=None,
        criterion=None,
        optimizer=None,
        scheduler=None,
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
