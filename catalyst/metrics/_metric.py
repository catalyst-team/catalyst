from typing import Any, Dict, Iterable, Optional
from abc import ABC, abstractmethod
from collections import defaultdict

import torch


class IMetric(ABC):
    """Interface for all Metrics.

    Args:
        compute_on_call: Computes and returns metric value during metric call.
            Used for per-batch logging. default: True
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

    def __call__(self, *args, **kwargs) -> Any:  # noqa: CCE001
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
    """@TODO: docs here"""

    def __init__(self, compute_on_call: bool = True, prefix: str = None, suffix: str = None):
        """@TODO: docs here"""
        super().__init__(compute_on_call=compute_on_call)
        self.prefix = prefix or ""
        self.suffix = suffix or ""

    @abstractmethod
    def update_key_value(self, *args, **kwargs) -> Dict[str, float]:
        """@TODO: docs here"""
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
    """Interface for all Metrics.

    Args:
        compute_on_call: @TODO: docs
        prefix:  @TODO: docs
        suffix:  @TODO: docs
    """

    def __init__(self, compute_on_call: bool = True, prefix: str = None, suffix: str = None):
        """Init."""
        super().__init__(compute_on_call=compute_on_call)
        self.prefix = prefix or ""
        self.suffix = suffix or ""

    @abstractmethod
    def reset(self, num_batches, num_samples) -> None:
        """Resets the metric to it's initial state.

        By default, this is called at the start of each loader
        (`on_loader_start` event).

        Args:
            num_batches: @TODO: docs.
            num_samples: @TODO: docs.
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


class AccumulationMetric(ICallbackLoaderMetric):
    """This metric accumulates all the input data along loader

    Args:
        accumulative_fields: list of keys to accumulate data from batch
        compute_on_call: if True, allows compute metric's value on call
        prefix: metric prefix
        suffix: metric suffix
    """

    def __init__(
        self,
        accumulative_fields: Iterable[str] = None,
        compute_on_call: bool = True,
        prefix: Optional[str] = None,
        suffix: Optional[str] = None,
    ) -> None:
        """Init AccumulationMetric"""
        super().__init__(compute_on_call=compute_on_call, prefix=prefix, suffix=suffix)
        self.accumulative_fields = accumulative_fields or ()
        self.storage = None
        self.num_samples = None
        self.collected_batches = None
        self.collected_samples = None

    def reset(self, num_batches: int, num_samples: int) -> None:
        """
        Reset metrics fields

        Args:
            num_batches: expected number of batches
            num_samples: expected number of samples to accumulate
        """
        self.num_samples = num_samples
        self.collected_batches = 0
        self.collected_samples = 0
        self.storage = None

    def _allocate_memory(self, shape_type_dict: Dict[str, Any]) -> None:
        """
        Allocate memory for data accumulation

        Args:
            shape_type_dict: dict that contains information about shape of each tensor and
                it's dtype
        """
        self.storage = defaultdict(torch.Tensor)
        for key in shape_type_dict:
            self.storage[key] = torch.empty(
                size=shape_type_dict[key]["shape"], dtype=shape_type_dict[key]["dtype"],
            )

    def update(self, **kwargs) -> None:
        """
        Update accumulated data with new batch

        Args:
            **kwargs: tensors that should be accumulates
        """
        if self.collected_batches == 0:
            shape_type_dict = {}
            for field_name in self.accumulative_fields:
                shape_type_dict[field_name] = {}
                shape_type_dict[field_name]["shape"] = (
                    self.num_samples,
                    *(kwargs[field_name].shape[1:]),
                )
                shape_type_dict[field_name]["dtype"] = kwargs[field_name].dtype
            self._allocate_memory(shape_type_dict=shape_type_dict)
        bs = 0
        for field_name in self.accumulative_fields:
            bs = kwargs[field_name].shape[0]
            self.storage[field_name][self.collected_samples : self.collected_samples + bs, ...] = (
                kwargs[field_name].detach().cpu()
            )
        self.collected_samples += bs
        self.collected_batches += 1

    def compute(self) -> Dict[str, torch.Tensor]:
        """
        Return accumulated data

        Returns:
            dict of accumulated data
        """
        return self.storage

    def compute_key_value(self) -> Dict[str, torch.Tensor]:
        """
        Return accumulated data

        Returns:
            dict of accumulated data
        """
        return self.compute()


__all__ = ["IMetric", "ICallbackBatchMetric", "ICallbackLoaderMetric", "AccumulationMetric"]
