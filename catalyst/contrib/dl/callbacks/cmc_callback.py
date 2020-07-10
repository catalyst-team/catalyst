from typing import List

import torch

from catalyst.core import IRunner
from catalyst.core.callback import CallbackOrder
from catalyst.dl import Callback
from catalyst.utils.metrics.cmc_score import cmc_score


class CMCScoreCallback(Callback):
    """
    Cumulative Matching Characteristics


    """

    def __init__(
        self,
        embeddings_key: str = "embeddings",
        labels_key: str = "labels",
        is_query_key: str = "query",
        prefix: str = "cmc",
        topk_args: List[int] = None,
    ):
        """
        This callback was designed to count
        cumulative matching characteristics.
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
        self._query_embeddings: torch.Tensor = None
        self._gallery_labels: torch.Tensor = None
        self._query_labels: torch.Tensor = None
        super().__init__(order=CallbackOrder.Metric)
        self._first_epoch = True
        self._gallery_idx = None
        self._query_idx = None

    def on_batch_end(self, runner: "IRunner"):
        """On batch end action"""
        query_mask = runner.input[self.is_query_key]
        # bool mask
        gallery_mask = ~query_mask
        query_embeddings = runner.output[self.embeddings_key][query_mask].cpu()
        gallery_embeddings = runner.output[self.embeddings_key][
            gallery_mask
        ].cpu()
        query_labels = runner.input[self.labels_key][query_mask].cpu()
        gallery_labels = runner.input[self.labels_key][gallery_mask].cpu()
        if self._first_epoch:
            self._accumulate_first_batch(
                query_embeddings,
                gallery_embeddings,
                query_labels,
                gallery_labels,
            )
        else:
            self._accumulate(
                query_embeddings,
                gallery_embeddings,
                query_labels,
                gallery_labels,
            )

    def _accumulate_first_batch(
        self,
        query_embeddings: torch.Tensor,
        gallery_embeddings: torch.Tensor,
        query_labels: torch.LongTensor,
        gallery_labels: torch.LongTensor,
    ) -> None:
        if self._query_embeddings is None:
            self._query_embeddings = query_embeddings
            self._query_labels = query_labels
        else:
            self._query_embeddings = torch.cat(
                (self._query_embeddings, query_embeddings), dim=0
            )
            self._query_labels = torch.cat(
                (self._query_labels, query_labels), dim=0
            )

        if self._gallery_embeddings is None:
            self._gallery_embeddings = gallery_embeddings
            self._gallery_labels = gallery_labels
        else:
            self._gallery_embeddings = torch.cat(
                (self._gallery_embeddings, gallery_embeddings), dim=0
            )
            self._gallery_labels = torch.cat(
                (self._gallery_labels, gallery_labels), dim=0
            )

    def _accumulate(
        self,
        query_embeddings: torch.Tensor,
        gallery_embeddings: torch.Tensor,
        query_labels: torch.LongTensor,
        gallery_labels: torch.LongTensor,
    ) -> None:
        if query_embeddings.shape[0] > 0:
            add_mask = self._query_idx + torch.arange(
                query_embeddings.shape[0]
            )
            self._query_embeddings[add_mask] = query_embeddings
            self._query_labels[add_mask] = query_labels
            self._query_idx += query_embeddings.shape[0]
        if gallery_embeddings.shape[0] > 0:
            add_mask = self._gallery_idx + torch.arange(
                gallery_embeddings.shape[0]
            )
            self._gallery_embeddings[add_mask] = gallery_embeddings
            self._gallery_labels[add_mask] = gallery_labels
            self._gallery_idx += gallery_embeddings.shape[0]

    def on_loader_end(self, runner: "IRunner"):
        """On loader end action"""
        conformity_matrix = self._query_labels.T == self._gallery_labels
        for k in self.list_args:
            metric = self._metric_fn(
                self._gallery_embeddings,
                self._query_embeddings,
                conformity_matrix,
                k,
            )
            runner.loader_metrics[f"{self._prefix}_{k}"] = metric
        self._first_epoch = False
        self._gallery_idx = 0
        self._query_idx = 0
