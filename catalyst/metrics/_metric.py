from typing import Any, Dict
from abc import ABC, abstractmethod


class IMetric(ABC):
    """Interface for all Metrics.

    Args:
        compute_on_call: Computes and returns metric value during metric call.
            Used for per-batch logging.
            default: ``True``
    """

    def __init__(self, compute_on_call: bool = True):
        """Interface for all Metrics."""
        self.compute_on_call = compute_on_call

    @abstractmethod
    def reset(self) -> None:
        """Resets the metric to it's initial state.

        By default, this is called at the start of each loader
        (`on_loader_start` event).
        """
        pass

    @abstractmethod
    def update(self, *args, **kwargs) -> Any:
        """Updates the metrics state using the passed data.

        By default, this is called at the end of each batch
        (`on_batch_end` event).

        Args:
            *args: some args :)
            **kwargs: some kwargs ;)
        """
        pass

    @abstractmethod
    def compute(self) -> Any:
        """Computes the metric based on it's accumulated state.

        By default, this is called at the end of each loader
        (`on_loader_end` event).

        Returns:
            Any: computed value, # noqa: DAR202
            it's better to return key-value
        """
        pass

    def __call__(self, *args, **kwargs) -> Any:
        """Computes the metric based on it's accumulated state.

        By default, this is called at the end of each batch
        (`on_batch_end` event).
        Returns computed value if `compute_on_call=True`.

        Returns:
            Any: computed value, it's better to return key-value.
        """
        value = self.update(*args, **kwargs)
        return self.compute() if self.compute_on_call else value


class ICallbackBatchMetric(IMetric):
    """Interface for all batch-based Metrics."""

    def __init__(self, compute_on_call: bool = True, prefix: str = None, suffix: str = None):
        """Init"""
        super().__init__(compute_on_call=compute_on_call)
        self.prefix = prefix or ""
        self.suffix = suffix or ""

    @abstractmethod
    def update_key_value(self, *args, **kwargs) -> Dict[str, float]:
        """Updates the metric based with new input.

        By default, this is called at the end of each loader
        (`on_loader_end` event).

        Args:
            *args: some args
            **kwargs: some kwargs

        Returns:
            Dict: computed value in key-value format.  # noqa: DAR202
        """
        pass

    @abstractmethod
    def compute_key_value(self) -> Dict[str, float]:
        """Computes the metric based on it's accumulated state.

        By default, this is called at the end of each loader
        (`on_loader_end` event).

        Returns:
            Dict: computed value in key-value format.  # noqa: DAR202
        """
        pass


class ICallbackLoaderMetric(IMetric):
    """Interface for all loader-based Metrics.

    Args:
        compute_on_call: Computes and returns metric value during metric call.
            Used for per-batch logging.
            default: ``True``
        prefix:  metrics prefix
        suffix:  metrics suffix
    """

    def __init__(self, compute_on_call: bool = True, prefix: str = None, suffix: str = None):
        """Init."""
        super().__init__(compute_on_call=compute_on_call)
        self.prefix = prefix or ""
        self.suffix = suffix or ""

    @abstractmethod
    def reset(self, num_batches: int, num_samples: int) -> None:
        """Resets the metric to it's initial state.

        By default, this is called at the start of each loader
        (`on_loader_start` event).

        Args:
            num_batches: number of expected batches.
            num_samples: number of expected samples.
        """
        pass

    @abstractmethod
    def update(self, *args, **kwargs) -> None:
        """Updates the metrics state using the passed data.

        By default, this is called at the end of each batch
        (`on_batch_end` event).

        Args:
            *args: some args :)
            **kwargs: some kwargs ;)
        """
        pass

    @abstractmethod
    def compute_key_value(self) -> Dict[str, float]:
        """Computes the metric based on it's accumulated state.

        By default, this is called at the end of each loader
        (`on_loader_end` event).

        Returns:
            Dict: computed value in key-value format.  # noqa: DAR202
        """
        # @TODO: could be refactored - we need custom exception here
        # we need this method only for callback metric logging
        pass


__all__ = ["IMetric", "ICallbackBatchMetric", "ICallbackLoaderMetric"]
