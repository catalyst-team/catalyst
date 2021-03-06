from typing import Any, Dict, Iterable, List, Optional
from collections import defaultdict

import torch

from catalyst.metrics import ICallbackLoaderMetric
from catalyst.metrics.functional._cmc_score import cmc_score


class AccumulationMetric(ICallbackLoaderMetric):
    """This metric accumulates all the input data along loader"""

    def __init__(
        self,
        compute_on_call: bool = True,
        prefix: Optional[str] = None,
        suffix: Optional[str] = None,
        accumulative_fields: Iterable[str] = (),
    ) -> None:
        """
        Init AccumulationMetric

        Args:
            compute_on_call: if True, allows compute metric's value on call
            prefix: metric's prefix
            suffix: metric's suffix
            accumulative_fields: list of keys to accumulate data from batch
        """
        super().__init__(
            compute_on_call=compute_on_call, prefix=prefix, suffix=suffix
        )
        self.accumulative_fields = accumulative_fields
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
                size=shape_type_dict[key]["shape"],
                dtype=shape_type_dict[key]["dtype"],
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
            self.storage[field_name][
                self.collected_samples : self.collected_samples + bs, ...
            ] = (kwargs[field_name].detach().cpu())
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


class CMCMetric(AccumulationMetric):
    """Cumulative Matching Characteristics"""

    def __init__(
        self,
        compute_on_call: bool = True,
        prefix: Optional[str] = None,
        suffix: Optional[str] = None,
        topk_args: Iterable[int] = None,
        embeddings_key: str = "features",
        labels_key: str = "targets",
        is_query_key: str = "is_query",
    ) -> None:
        """
        Init CMCMetric

        Args:
            compute_on_call: if True, allows compute metric's value on call
            prefix: metric's prefix
            suffix: metric's suffix
            embeddings_key: key of embedding tensor in batch
            labels_key: key of label tensor in batch
            is_query_key: key of query flag tensor in batch
            topk_args: list of k, specifies which cmc@k should be calculated
        """
        super().__init__(
            compute_on_call=compute_on_call,
            prefix=prefix,
            suffix=suffix,
            accumulative_fields=[embeddings_key, labels_key, is_query_key],
        )
        self.embeddings_key = embeddings_key
        self.labels_key = labels_key
        self.is_query_key = is_query_key
        self.topk_args = topk_args or (1, )
        self.metric_name = f"{self.prefix}cmc{self.suffix}"

    def compute(self) -> List[float]:
        """
        Compute cmc@k metrics with all the accumulated data for all k.

        Returns:
            list of metrics values
        """
        query_mask = (self.storage[self.is_query_key] == 1).to(torch.bool)

        embeddings = self.storage[self.embeddings_key].float()
        labels = self.storage[self.labels_key]

        query_embeddings = embeddings[query_mask]
        query_labels = labels[query_mask]

        gallery_embeddings = embeddings[~query_mask]
        gallery_labels = labels[~query_mask]

        conformity_matrix = (gallery_labels == query_labels.reshape(-1, 1)).to(
            torch.bool
        )

        metrics = []
        for k in self.topk_args:
            value = cmc_score(
                query_embeddings=query_embeddings,
                gallery_embeddings=gallery_embeddings,
                conformity_matrix=conformity_matrix,
                topk=k,
            )
            metrics.append(value)

        return metrics

    def compute_key_value(self) -> Dict[str, float]:
        """
        Compute cmc@k metrics with all the accumulated data for all k.

        Returns:
            metrics values in key-value format
        """
        values = self.compute()
        kv_metrics = {
            f"{self.metric_name}{k:02d}": value
            for k, value in zip(self.topk_args, values)
        }
        return kv_metrics


__all__ = [
    "AccumulationMetric",
    "CMCMetric",
]
