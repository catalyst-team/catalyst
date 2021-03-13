from typing import Any, Dict
from abc import ABC, abstractmethod
from contextlib import contextmanager

from catalyst.typing import Criterion, Model, Optimizer, Scheduler


@contextmanager
def nullcontext(enter_result=None):
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
    """

    # @property
    # @abstractmethod
    # def device(self) -> Device:
    #     pass

    @property
    @abstractmethod
    def rank(self) -> int:
        """"Process rank for distributed training."""
        pass

    @property
    @abstractmethod
    def world_size(self) -> int:
        """Process world size  for distributed training."""
        # only for ddp
        pass

    @property
    def is_ddp(self) -> bool:
        """Boolean flag for distributed run."""
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
        """Moves ``tensor_or_module`` to Engine's deivce.

        Args:
            tensor: tensor to sync across the processes.
            mode: tensor synchronization type,
                should be one of 'sum' or 'mean'.
                Default is 'mean'.
        """
        pass

    @abstractmethod
    def sync_tensor(self, tensor: Any, mode: str) -> Any:
        """Syncs ``tensor`` over ``world_size`` in distributed mode."""
        pass

    @abstractmethod
    def init_components(
        self, model_fn=None, criterion_fn=None, optimizer_fn=None, scheduler_fn=None,
    ):
        """Inits the runs components."""
        pass

    @abstractmethod
    def deinit_components(self):
        """Deinits the runs components."""
        # only for ddp
        pass

    @abstractmethod
    def zero_grad(self, loss, model, optimizer) -> None:
        """Abstraction over ``model.zero_grad()`` step."""
        pass

    @abstractmethod
    def backward_loss(self, loss, model, optimizer) -> None:
        """Abstraction over ``loss.backward()`` step."""
        pass

    @abstractmethod
    def optimizer_step(self, loss, model, optimizer) -> None:
        """Abstraction over ``optimizer.step()`` step."""
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

        Returns:
            torch-based checkpoint with ``model_state_dict``,
            ``criterion_state_dict``, ``optimizer_state_dict``,
            ``scheduler_state_dict`` keys.
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
        """AMP scaling context. Default autocast context does not scale anything.

        Args:
            *args: some args
            **kwargs: some kwargs

        Returns:
            context
        """
        return nullcontext()
