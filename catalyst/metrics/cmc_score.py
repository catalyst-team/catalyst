from collections import defaultdict
from typing import Dict, Any, Optional, Iterable, Union, List

import torch

from catalyst.metrics.functional.cmc_score import cmc_score
from catalyst.metrics import ICallbackLoaderMetric


class AccumulationMetric(ICallbackLoaderMetric):
    def __init__(
            self,
            compute_on_call: bool = True,
            accumulative_fields: Iterable[str] = (),
    ):
        super().__init__(compute_on_call=compute_on_call)
        self.accumulative_fields = accumulative_fields
        self.storage = None
        self.num_samples = None
        self.collected_batches = None
        self.tmp_idx = None

    def reset(self, num_batches: int, num_samples: int) -> None:
        self.num_samples = num_samples
        self.collected_batches = 0
        self.tmp_idx = 0
        self.storage = None

    def _allocate_memory(self, shape_type_dict: Dict[str, Any]):
        self.storage = defaultdict(torch.Tensor)
        for key in shape_type_dict:
            self.storage[key] = torch.empty(
                size=shape_type_dict[key]["shape"], dtype=shape_type_dict[key]["dtype"]
            )

    def update(self, batch) -> None:
        if self.collected_batches == 0:
            shape_type_dict = {}
            for field_name in self.accumulative_fields:
                shape_type_dict[field_name] = {}
                shape_type_dict[field_name]["shape"] = (
                    self.num_samples, *(batch[field_name].shape[1:])
                )
                shape_type_dict[field_name]["dtype"] = batch[field_name].dtype
            self._allocate_memory(shape_type_dict=shape_type_dict)
        bs = 0
        for field_name in self.accumulative_fields:
            bs = batch[field_name].shape[0]
            self.storage[
                field_name
            ][self.tmp_idx: self.tmp_idx + bs, ...] = batch[field_name].detach().cpu()
        self.tmp_idx += bs
        self.collected_batches += 1

    def compute(self) -> Any:
        return self.storage

    def compute_key_value(self) -> Dict[str, float]:
        return self.compute()


class CMCMetric(AccumulationMetric):
    def __init__(
            self,
            compute_on_call: bool = True,
            embeddings_key: str = "features",
            labels_key: str = "targets",
            is_query_key: str = "is_query",
            topk_args: Iterable[int] = (1, ),
    ):
        super().__init__(
            compute_on_call=compute_on_call,
            accumulative_fields=[embeddings_key, labels_key, is_query_key]
        )
        self.embeddings_key = embeddings_key
        self.labels_key = labels_key
        self.is_query_key = is_query_key
        self.topk_args = topk_args
        self.metric_name = "cmc"

    def compute(self) -> List[float]:
        query_embeddings = self.storage[self.embeddings_key][self.storage[self.is_query_key] == 1]
        query_labels = self.storage[self.labels_key][self.storage[self.is_query_key] == 1]

        gallery_embeddings = self.storage[self.embeddings_key][self.storage[self.is_query_key] == 0]
        gallery_labels = self.storage[self.labels_key][self.storage[self.is_query_key] == 0]

        conformity_matrix = (gallery_labels == query_labels.reshape(-1, 1)).bool()

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
        values = self.compute()
        kv_metrics = {
            f"{self.metric_name}{k:02d}": value for k, value in zip(self.topk_args, values)
        }
        return kv_metrics


__all__ = ["CMCMetric", ]
