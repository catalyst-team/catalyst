from typing import Any, Dict
from abc import ABC, abstractmethod
from contextlib import contextmanager

from catalyst.typing import Device


@contextmanager
def nullcontext(enter_result=None):
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
        pass

    @property
    @abstractmethod
    def world_size(self) -> int:
        # only for ddp
        pass

    @abstractmethod
    def sync_device(self, tensor_or_module: Any) -> Any:
        pass
        # return any2device(batch, self.device)

    @abstractmethod
    def sync_tensor(self, tensor: Any) -> Any:
        pass

    @abstractmethod
    def init_components(
        self, model_fn=None, criterion_fn=None, optimizer_fn=None, scheduler_fn=None,
    ):
        pass

    @abstractmethod
    def deinit_components(self):
        # only for ddp
        pass

    @abstractmethod
    def zero_grad(self, loss, model, optimizer) -> None:
        pass

    @abstractmethod
    def backward_loss(self, loss, model, optimizer) -> None:
        pass

    @abstractmethod
    def optimizer_step(self, loss, model, optimizer) -> None:
        pass

    @abstractmethod
    def pack_checkpoint(
        self, model=None, criterion=None, optimizer=None, scheduler=None, **kwargs,
    ) -> Dict:
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
        pass

    @abstractmethod
    def save_checkpoint(self, checkpoint: Dict, path: str) -> None:
        pass

    @abstractmethod
    def load_checkpoint(self, path: str) -> Dict:
        pass

    def autocast(self, *args, **kwargs):
        """AMP scaling context.
        Default autocast context does not scale anything.
        """
        return nullcontext()
