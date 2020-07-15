from typing import List

import torch

from catalyst.core import IRunner
from catalyst.core.callback import CallbackOrder
from catalyst.dl import Callback
from catalyst.utils.metrics.cmc_score import cmc_score


class CMCScoreCallback(Callback):
    """
    Cumulative Matching Characteristics callback

    You should use it with `ControlFlowCallback`
    and add all query/gallery sets to loaders.
    Loaders should contain "is_query" and "label" key.

    .. code-block:: python

        runner.train(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            loaders=loaders,
            logdir="./logdir",
            num_epochs=5,
            callbacks=[
                ControlFlowCallback(
                    base_callback=CMCScoreCallback(
                        topk_args=[1, 5]
                    ),
                    loaders=["split_1", "split_2"]
                )
            ]
        )
    """

    def __init__(
        self,
        embeddings_key: str = "logits",
        labels_key: str = "targets",
        is_query_key: str = "is_query",
        prefix: str = "cmc",
        topk_args: List[int] = None,
    ):
        """
        This callback was designed to count
        cumulative matching characteristics.
        If current object is from query your dataset
        should output `True` in `is_query_key`
        and false if current object is from gallery.
        You can see `QueryGalleryDataset` in
        `catalyst.contrib.data.ml` for more information.
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
        self._gallery_idx = None
        self._query_idx = None
        self._query_size = None
        self._gallery_size = None

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

        if self._query_embeddings is None:
            emb_dim = query_embeddings.shape[1]
            self._query_embeddings = torch.empty(self._query_size, emb_dim)
            self._gallery_embeddings = torch.empty(self._gallery_size, emb_dim)
        self._accumulate(
            query_embeddings, gallery_embeddings, query_labels, gallery_labels,
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

    def on_loader_start(self, runner: "IRunner"):
        """On loader start action"""
        loader = runner.loaders[runner.loader_name]
        self._query_size = loader.dataset.query_size
        self._gallery_size = loader.dataset.gallery_size
        self._query_labels = torch.empty(self._query_size, dtype=torch.long)
        self._gallery_labels = torch.empty(
            self._gallery_size, dtype=torch.long
        )
        self._gallery_idx = 0
        self._query_idx = 0

    def on_loader_end(self, runner: "IRunner"):
        """On loader end action"""
        conformity_matrix = self._query_labels == self._gallery_labels.reshape(
            -1, 1
        )
        for k in self.list_args:
            metric = self._metric_fn(
                self._gallery_embeddings,
                self._query_embeddings,
                conformity_matrix,
                k,
            )
            runner.epoch_metrics[
                f"{runner.loader_name}_{self._prefix}_{k}"
            ] = metric
        self._gallery_embeddings = None
        self._query_embeddings = None
