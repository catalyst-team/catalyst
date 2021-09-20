from typing import Any, Dict, Iterable, Optional
from collections import defaultdict

import torch

from catalyst.metrics._metric import ICallbackLoaderMetric


class AccumulativeMetric(ICallbackLoaderMetric):
    """This metric accumulates all the input data along loader

    Args:
        keys: list of keys to accumulate data from batch
        compute_on_call: if True, allows compute metric's value on call
        prefix: metric prefix
        suffix: metric suffix
    """

    def __init__(
        self,
        keys: Iterable[str] = None,
        compute_on_call: bool = True,
        prefix: Optional[str] = None,
        suffix: Optional[str] = None,
    ) -> None:
        """Init AccumulativeMetric"""
        super().__init__(compute_on_call=compute_on_call, prefix=prefix, suffix=suffix)
        self.keys = keys or ()
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
                size=shape_type_dict[key]["shape"], dtype=shape_type_dict[key]["dtype"]
            )

    def update(self, **kwargs) -> None:
        """
        Update accumulated data with new batch

        Args:
            **kwargs: tensors that should be accumulates
        """
        if self.collected_batches == 0:
            shape_type_dict = {}
            for field_name in self.keys:
                shape_type_dict[field_name] = {}
                shape_type_dict[field_name]["shape"] = (
                    self.num_samples,
                    *(kwargs[field_name].shape[1:]),
                )
                shape_type_dict[field_name]["dtype"] = kwargs[field_name].dtype
            self._allocate_memory(shape_type_dict=shape_type_dict)
        bs = 0
        for field_name in self.keys:
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


__all__ = ["AccumulativeMetric"]
