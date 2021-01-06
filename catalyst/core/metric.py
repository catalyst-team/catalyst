from typing import Any, Dict
from abc import ABC, abstractmethod


# origin:
# https://github.com/catalyst-team/catalyst/blob/master/catalyst/tools/meters/meter.py
class IMetric(ABC):
    def __init__(self, compute_on_call: bool = True):
        """Interface for all Metrics.

        Args:
            compute_on_call:
                Computes and returns metric value during metric call.
                Used for per-batch logging. default: True
        """
        self.compute_on_call = compute_on_call

    @abstractmethod
    def reset(self) -> None:
        """Resets the metric to it's initial state.

        By default, this is called at the start of each loader
        (`on_loader_start` event).
        """
        pass

    @abstractmethod
    def update(self, *args, **kwargs) -> None:
        """Updates the metric's state using the passed data.

        By default, this is called at the end of each batch
        (`on_batch_end` event).
        """
        pass

    @abstractmethod
    def compute(self) -> Any:
        """Computes the metric based on it's accumulated state.

        By default, this is called at the end of each loader
        (`on_loader_end` event).

        Returns:
            Any: it's better to return dict-like

        Raises:
            NotImplementedError: raised when the metric cannot be computed.
        """
        pass

    @abstractmethod
    def get_dict_value(self) -> Dict:
        pass

    def __call__(self, *args, **kwargs) -> Any:
        pass
