from typing import List

import torch

from catalyst.contrib.datasets.metric_learning import QueryGalleryDataset
from catalyst.core import IRunner
from catalyst.core.callback import CallbackOrder
from catalyst.dl import Callback
from catalyst.dl.callbacks.metrics.functional import get_default_topk_args
from catalyst.utils.metrics.cmc_score import cmc_score

TORCH_BOOL = torch.bool if torch.__version__ > "1.1.0" else torch.ByteTensor


class CMCScoreCallback(Callback):
    """
    Cumulative Matching Characteristics callback

    You should use it with `ControlFlowCallback`
    and add all query/gallery sets to loaders.
    Loaders should contain "is_query" and "label" key.

    An usage example can be found in Readme.md:
    "CV - MNIST with Metric Learning".
    Or you can also found full metric learning pipeline
    :ref:`here <catalyst.test._tests_scripts.dl_z_mvp_mnist_metric_learning>`.
    """

    def __init__(
        self,
        embeddings_key: str = "logits",
        labels_key: str = "targets",
        is_query_key: str = "is_query",
        prefix: str = "cmc",
        topk_args: List[int] = None,
        num_classes: int = None,
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
            num_classes (int): number of classes to calculate ``accuracy_args``
                if ``topk_args`` is None

        """
        super().__init__(order=CallbackOrder.Metric)
        self.list_args = topk_args or get_default_topk_args(num_classes)
        self._metric_fn = cmc_score
        self._prefix = prefix
        self.embeddings_key = embeddings_key
        self.labels_key = labels_key
        self.is_query_key = is_query_key
        self._gallery_embeddings: torch.Tensor = None
        self._query_embeddings: torch.Tensor = None
        self._gallery_labels: torch.Tensor = None
        self._query_labels: torch.Tensor = None
        self._gallery_idx = None
        self._query_idx = None
        self._query_size = None
        self._gallery_size = None

    def _accumulate(
        self,
        query_embeddings: torch.Tensor,
        gallery_embeddings: torch.Tensor,
        query_labels: torch.LongTensor,
        gallery_labels: torch.LongTensor,
    ) -> None:
        if query_embeddings.shape[0] > 0:
            add_indices = self._query_idx + torch.arange(
                query_embeddings.shape[0]
            )
            self._query_embeddings[add_indices] = query_embeddings
            self._query_labels[add_indices] = query_labels
            self._query_idx += query_embeddings.shape[0]
        if gallery_embeddings.shape[0] > 0:
            add_indices = self._gallery_idx + torch.arange(
                gallery_embeddings.shape[0]
            )
            self._gallery_embeddings[add_indices] = gallery_embeddings
            self._gallery_labels[add_indices] = gallery_labels
            self._gallery_idx += gallery_embeddings.shape[0]

    def on_batch_end(self, runner: "IRunner"):
        """On batch end action"""
        query_mask = runner.input[self.is_query_key]
        # bool mask
        query_mask = query_mask.type(TORCH_BOOL)
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

    def on_loader_start(self, runner: "IRunner"):
        """On loader start action"""
        assert isinstance(
            runner.loaders[runner.loader_name].dataset, QueryGalleryDataset
        )
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
        assert (
            self._gallery_idx == self._gallery_size
        ), "An error occurred during the accumulation process."

        assert (
            self._query_idx == self._query_size
        ), "An error occurred during the accumulation process."

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
            runner.loader_metrics[f"{self._prefix}_{k}"] = metric
            runner.epoch_metrics[
                f"{runner.loader_name}_{self._prefix}_{k}"
            ] = metric
        self._gallery_embeddings = None
        self._query_embeddings = None
