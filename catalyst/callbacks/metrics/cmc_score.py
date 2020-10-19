from typing import List, TYPE_CHECKING

import torch

from catalyst.core.callback import Callback, CallbackOrder
from catalyst.data.dataset.metric_learning import QueryGalleryDataset
from catalyst.metrics.cmc_score import cmc_score, masked_cmc_score
from catalyst.metrics.functional import get_default_topk_args

if TYPE_CHECKING:
    from catalyst.core.runner import IRunner

TORCH_BOOL = torch.bool if torch.__version__ > "1.1.0" else torch.ByteTensor


class CMCScoreCallback(Callback):
    """
    Cumulative Matching Characteristics callback.

    .. note::

        You should use it with `ControlFlowCallback`
        and add all query/gallery sets to loaders.
        Loaders should contain "is_query" and "label" key.

    An usage example can be found in Readme.md under
    "CV - MNIST with Metric Learning".
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
        `catalyst.contrib.datasets.metric_learning` for more information.
        On batch end callback accumulate all embeddings

        Args:
            embeddings_key: embeddings key in output dict
            labels_key: labels key in output dict
            is_query_key: bool key True if current
                object is from query
            prefix: key for the metric's name
            topk_args: specifies which cmc@K to log.
                [1] - cmc@1
                [1, 3] - cmc@1 and cmc@3
                [1, 3, 5] - cmc@1, cmc@3 and cmc@5
            num_classes: number of classes to calculate ``accuracy_args``
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
        dataset = runner.loaders[runner.loader_name].dataset
        assert isinstance(dataset, QueryGalleryDataset)
        self._query_size = dataset.query_size
        self._gallery_size = dataset.gallery_size
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

        conformity_matrix = self._gallery_labels == self._query_labels.reshape(
            -1, 1
        )
        for key in self.list_args:
            metric = self._metric_fn(
                query_embeddings=self._query_embeddings,
                gallery_embeddings=self._gallery_embeddings,
                conformity_matrix=conformity_matrix,
                topk=key,
            )
            runner.loader_metrics[f"{self._prefix}{key:02}"] = metric
        self._gallery_embeddings = None
        self._query_embeddings = None


class ReidCMCScoreCallback(Callback):
    def __init__(
        self,
        embeddings_key: str = "logits",
        labels_key: str = "targets",
        is_query_key: str = "is_query",
        person_key: str = "pid",
        camera_key: str = "cam_id",
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
        `catalyst.contrib.datasets.metric_learning` for more information.
        On batch end callback accumulate all embeddings

        Args:
            embeddings_key: embeddings key in output dict
            labels_key: labels key in output dict
            is_query_key: bool key True if current
                object is from query
            prefix: key for the metric's name
            topk_args: specifies which cmc@K to log.
                [1] - cmc@1
                [1, 3] - cmc@1 and cmc@3
                [1, 3, 5] - cmc@1, cmc@3 and cmc@5
            num_classes: number of classes to calculate ``accuracy_args``
                if ``topk_args`` is None

        """
        super().__init__(order=CallbackOrder.Metric)
        self.list_args = topk_args or get_default_topk_args(num_classes)
        self._metric_fn = masked_cmc_score
        self._prefix = prefix
        self.embeddings_key = embeddings_key
        self.labels_key = labels_key
        self.is_query_key = is_query_key
        self.person_key = person_key
        self.camera_key = camera_key
        self._gallery_embeddings: torch.Tensor = None
        self._query_embeddings: torch.Tensor = None
        self._gallery_labels: torch.Tensor = None
        self._query_labels: torch.Tensor = None
        self._query_cameras: torch.Tensor = None
        self._gallery_cameras: torch.Tensor = None
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
        query_cameras: torch.Tensor,
        gallery_cameras: torch.Tensor,
    ) -> None:
        if query_embeddings.shape[0] > 0:
            add_indices = self._query_idx + torch.arange(
                query_embeddings.shape[0]
            )
            self._query_embeddings[add_indices] = query_embeddings
            self._query_labels[add_indices] = query_labels
            self._query_cameras[add_indices] = query_cameras
            self._query_idx += query_embeddings.shape[0]
        if gallery_embeddings.shape[0] > 0:
            add_indices = self._gallery_idx + torch.arange(
                gallery_embeddings.shape[0]
            )
            self._gallery_embeddings[add_indices] = gallery_embeddings
            self._gallery_labels[add_indices] = gallery_labels
            self._gallery_cameras[add_indices] = gallery_cameras
            self._gallery_idx += gallery_embeddings.shape[0]

    def on_batch_end(self, runner: "IRunner"):
        """On batch end action"""
        query_mask = runner.input[self.is_query_key]
        # bool mask
        query_mask = query_mask.type(TORCH_BOOL)
        gallery_mask = ~query_mask

        embeddings = runner.output[self.embeddings_key]
        labels = runner.input[self.labels_key][self.person_key]
        cameras = runner.input[self.labels_key][self.camera_key]

        query_embeddings = embeddings[query_mask].cpu()
        gallery_embeddings = embeddings[gallery_mask].cpu()

        query_labels = labels[query_mask].cpu()
        gallery_labels = labels[gallery_mask].cpu()

        query_cameras = cameras[query_mask].cpu()
        gallery_cameras = cameras[gallery_mask].cpu()

        if self._query_embeddings is None:
            emb_dim = query_embeddings.shape[1]
            self._query_embeddings = torch.empty(self._query_size, emb_dim)
            self._gallery_embeddings = torch.empty(self._gallery_size, emb_dim)
        self._accumulate(
            query_embeddings=query_embeddings,
            gallery_embeddings=gallery_embeddings,
            query_labels=query_labels,
            gallery_labels=gallery_labels,
            query_cameras=query_cameras,
            gallery_cameras=gallery_cameras,
        )

    def on_loader_start(self, runner: "IRunner"):
        """On loader start action"""
        dataset = runner.loaders[runner.loader_name].dataset
        assert isinstance(dataset, QueryGalleryDataset)
        self._query_size = dataset.query_size
        self._gallery_size = dataset.gallery_size
        self._query_labels = torch.empty(self._query_size, dtype=torch.long)
        self._gallery_labels = torch.empty(
            self._gallery_size, dtype=torch.long
        )
        self._query_cameras = torch.empty(self._query_size, dtype=torch.long)
        self._gallery_cameras = torch.empty(
            self._gallery_size, dtype=torch.long
        )
        self._gallery_idx = 0
        self._query_idx = 0

    @staticmethod
    def _get_conformity_matrix(
        gallery_labels: torch.Tensor, query_labels: torch.Tensor
    ) -> torch.Tensor:
        conformity_matrix = gallery_labels == query_labels.reshape(-1, 1)
        return conformity_matrix

    def on_loader_end(self, runner: "IRunner"):
        """On loader end action"""
        assert (
            self._gallery_idx == self._gallery_size
        ), "An error occurred during the accumulation process."

        assert (
            self._query_idx == self._query_size
        ), "An error occurred during the accumulation process."

        conformity_matrix = self._get_conformity_matrix(
            gallery_labels=self._gallery_labels,
            query_labels=self._query_labels,
        )
        cameras_conformity_matrix = self._get_conformity_matrix(
            gallery_labels=self._gallery_cameras,
            query_labels=self._query_cameras,
        )
        mask = ~(conformity_matrix * cameras_conformity_matrix)
        for key in self.list_args:
            metric = self._metric_fn(
                query_embeddings=self._query_embeddings,
                gallery_embeddings=self._gallery_embeddings,
                conformity_matrix=conformity_matrix,
                mask=mask,
                topk=key,
            )
            runner.loader_metrics[f"{self._prefix}{key:02}"] = metric
        self._gallery_embeddings = None
        self._query_embeddings = None


__all__ = ["CMCScoreCallback", "ReidCMCScoreCallback"]
