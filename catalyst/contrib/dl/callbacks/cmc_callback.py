from typing import List

import torch

from catalyst.dl import Callback
from catalyst.utils.metrics.cmc_score import cmc_score


class CMCScoreCallback(Callback):
    """Cumulative Matching Characteristics"""

    def __init__(
        self,
        embeddings_key: str = "features",
        labels_key: str = "labels",
        is_query_key: str = "query",
        prefix: str = "cmc",
        topk_args: List[int] = None,
    ):
        """
        This callback was designed to count cumulative matching characteristics.
        If current object is from query your dataset
        should output `True` in `is_query_key`
        and false if current object is from gallery.
        On batch end callback accumulate all embeddings
        Args:
            embeddings_key (str): embeddings key in output dict
            labels_key (str): labels key in output dict
            is_query_key (str): bool key True if current
                object is from query
            prefix (str): key for the metric's name
            topk_args (List[int]): specifies which cmc@K to log.
                [1] - cmc@1
                [1, 3] - cmc@1 and cmc@3
                [1, 3, 5] - cmc@1, cmc@3 and cmc@5

        """
        self.list_args = topk_args or [1]
        self._metric_fn = cmc_score
        self._prefix = prefix
        self.embeddings_key = embeddings_key
        self.labels_key = labels_key
        self.is_query_key = is_query_key
        self._gallery_embeddings: torch.Tensor = None
        self._queries_embeddings: torch.Tensor = None
        self._gallery_labels: torch.Tensor = None
        self._query_labels: torch.Tensor = None

    def on_batch_end(self, runner: "IRunner"):
        """On batch end action"""
        current_query_mask = runner.output[self.is_query_key]
        # bool mask
        current_gallery_mask = ~current_query_mask
        current_query_embeddings = runner.output[self.embeddings_key][
            current_query_mask
        ]
        current_gallery_embeddings = runner.output[self.embeddings_key][
            current_gallery_mask
        ]
        current_query_labels = runner.output[self.labels_key][
            current_query_mask
        ]
        current_gallery_labels = runner.output[self.labels_key][
            current_gallery_mask
        ]

        if self._queries_embeddings is None:
            self._queries_embeddings = current_query_embeddings
            self._query_labels = current_query_labels
        else:
            self._queries_embeddings = torch.cat(
                (self._queries_embeddings, current_query_embeddings), dim=0
            )
            self._query_labels = torch.cat(
                (self._queries_embeddings, current_query_labels), dim=0
            )

        if self._gallery_embeddings is None:
            self._gallery_embeddings = current_gallery_embeddings
            self._gallery_labels = current_gallery_labels
        else:
            self._gallery_embeddings = torch.cat(
                (self._gallery_embeddings, current_gallery_embeddings), dim=0
            )
            self._query_labels = torch.cat(
                (self._gallery_embeddings, current_gallery_labels), dim=0
            )

    def on_loader_end(self, runner: "IRunner"):
        """On loader end action"""
        conformity_matrix = self._query_labels == self._gallery_labels.T
        for k in self.list_args:
            metric = self._metric_fn(
                self._gallery_embeddings,
                self._queries_embeddings,
                conformity_matrix,
                k,
            )
            runner.loader_metrics[f"{self._prefix}_{k}"] = metric
