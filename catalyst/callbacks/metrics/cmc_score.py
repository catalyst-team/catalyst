from typing import Callable, Dict, List, TYPE_CHECKING, Union, Any
import operator

import torch

from catalyst.core.callback import Callback, CallbackOrder
from catalyst.data.dataset.metric_learning import QueryGalleryDataset
from catalyst.metrics.cmc_score import cmc_score, masked_cmc_score
from catalyst.metrics.functional import get_default_topk_args

if TYPE_CHECKING:
    from catalyst.core.runner import IRunner

TORCH_BOOL = torch.bool if torch.__version__ > "1.1.0" else torch.ByteTensor

DATA_SPLITS = ("query", "gallery")
FIELDS = ("embeddings", "labels")
REID_FIELDS = ("embeddings", "pids", "cids")
TExtractKey = Union[str, Callable[[Dict[str, Any]], torch.Tensor]]


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
        embeddings_key: TExtractKey = "logits",
        labels_key: TExtractKey = "targets",
        is_query_key: TExtractKey = "is_query",
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
            embeddings_key: embeddings key in output dict or a function that
                extracts embeddings from input dict
            labels_key: labels key in input dict or a function that
                extracts labels from input dict
            is_query_key: flag is_query key in input dict or a function that
                extracts is_query from input dict
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
        self._extract_functions = {
            name: operator.itemgetter(key) if isinstance(key, str) else key
            for name, key in (
                ("embeddings", embeddings_key),
                ("labels", labels_key),
                ("is_query", is_query_key),
            )
        }
        # accumulative embeddings can be extracted from runner.output
        # but other fields are contained in runner.input
        self._extract_source = {
            "embeddings": "output",
            "labels": "input",
            "is_query": "input",
        }
        self._fields = FIELDS
        # here we have {"query": {"embeddings": None, "labels": None}, ...}
        self._accumulative_fields = {
            data_split: {field: None for field in self._fields}
            for data_split in DATA_SPLITS
        }
        # how many samples there are in query and gallery sets now
        self._indices = {key: 0 for key in DATA_SPLITS}
        # how many samples for query and gallery we expect to get
        self._sizes = {key: None for key in DATA_SPLITS}

    def _accumulate_batch(
        self, data_split: str, data: Dict[str, torch.Tensor]
    ):
        """
        Accumulate data from query or gallery part of batch data
        Args:
            data_split: "query" or "gallery" -- shows if we should accumulate
                the data as query or gallery samples
            data: all the data that should be accumulated at the end of the
                batch; we except it to contain all the fields of
                the following callback (callback._fields)
        """
        assert data_split in DATA_SPLITS, \
            f"Unexpected data split: should be one of \"query\" or " \
            f"\"gallery\", got {data_split}"

        for field in self._fields:
            assert field in data, f"Data should contain {field}"

        n_features = data["embeddings"].shape[0]
        add_indices = self._indices[data_split] + torch.arange(n_features)
        for key, value in data.items():
            self._accumulative_fields[data_split][key][
                add_indices
            ] = value
        self._indices[data_split] += n_features

    def _get_batch_data(self, runner: "IRunner") -> Dict[str, torch.Tensor]:
        """
        Extract data for accumulative fields from batch
        Args:
            runner: runner of the experiment

        Returns:
            dict of extracted data
        """
        batch_data = {
            key: function(getattr(runner, self._extract_source[key]))
            for key, function in self._extract_functions.items()
        }
        return batch_data

    def on_batch_end(self, runner: "IRunner"):
        """On batch end action"""
        batch_data = self._get_batch_data(runner=runner)

        query_mask = batch_data["is_query"].type(TORCH_BOOL)
        gallery_mask = ~query_mask

        for data_split, mask in (
            ("query", query_mask), ("gallery", gallery_mask),
        ):
            embeddings = batch_data["embeddings"][mask].cpu()
            labels = batch_data["labels"][mask].cpu()

            # Now we got some embeddings and can init fields for embeddings
            # accumulation if they haven't been initialized yet
            if self._accumulative_fields[data_split]["embeddings"] is None:
                self._accumulative_fields[data_split]["embeddings"] = torch.empty(
                    size=(self._sizes[data_split], embeddings.shape[1])
                )

            self._accumulate_batch(
                data_split=data_split,
                data={"embeddings": embeddings, "labels": labels},
            )

    def _init_accumulative_fields(self):
        """Init fields for accumulation"""
        for data_split, size in self._sizes.items():
            for field in self._fields:
                self._accumulative_fields[data_split][field] = torch.empty(
                    size=size, dtype=torch.long
                )
            self._accumulative_fields[data_split]["embeddings"] = None

    def on_loader_start(self, runner: "IRunner"):
        """On loader start action"""
        dataset = runner.loaders[runner.loader_name].dataset
        assert isinstance(dataset, QueryGalleryDataset)
        self._sizes["query"] = dataset.query_size
        self._sizes["gallery"] = dataset.gallery_size
        self._init_accumulative_fields()

    @staticmethod
    def _get_conformity_matrix(
        query_labels: torch.Tensor, gallery_labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Count conformity matrix for query and gallery labels.
        Args:
            query_labels: list of labels of query samples
            gallery_labels: list of labels of gallery samples

        Returns:
            tensor of shape (query_size, gallery_size),
                conformity matrix[i][j] == 1 if i-th element from query_labels
                is equal to the j-th element of gallery_labels
        """
        conformity_matrix = gallery_labels == query_labels.reshape(-1, 1)
        return conformity_matrix

    def _check_completeness(self):
        """
        Check if we have accumulated all the samples for query and gallery
        """
        for data_split in DATA_SPLITS:
            assert (
                    self._indices[data_split] == self._sizes[data_split]
            ), f"An error occurred during the accumulation process: " \
               f"expected to get {self._sizes[data_split]} samples for" \
               f"{data_split}, got {self._indices[data_split]}"

    def _reset_fields(self):
        """Reset all the accumulated data"""
        for data_split in DATA_SPLITS:
            for field in self._fields:
                self._accumulative_fields[data_split][field] = None
        self._indices = {key: 0 for key in self._indices}
        self._sizes = {key: None for key in self._sizes}

    def on_loader_end(self, runner: "IRunner"):
        """On loader end action"""
        self._check_completeness()
        conformity_matrix = self._get_conformity_matrix(
            query_labels=self._accumulative_fields["query"]["labels"],
            gallery_labels=self._accumulative_fields["gallery"]["labels"],
        )
        for key in self.list_args:
            metric = self._metric_fn(
                query_embeddings=self._accumulative_fields["query"][
                    "embeddings"
                ],
                gallery_embeddings=self._accumulative_fields["gallery"][
                    "embeddings"
                ],
                conformity_matrix=conformity_matrix,
                topk=key,
            )
            runner.loader_metrics[f"{self._prefix}{key:02}"] = metric
        self._reset_fields()


class ReidCMCScoreCallback(CMCScoreCallback):
    def __init__(
        self,
        embeddings_key: TExtractKey = "logits",
        is_query_key: TExtractKey = "is_query",
        prefix: str = "cmc",
        topk_args: List[int] = None,
        num_classes: int = None,
        person_key: TExtractKey = "pid",
        camera_key: TExtractKey = "cid",
        **kwargs,
    ):
        """
        """
        super().__init__()
        self.list_args = topk_args or get_default_topk_args(num_classes)
        self._metric_fn = masked_cmc_score
        self._prefix = prefix
        self._extract_functions = {
            name: operator.itemgetter(key) if isinstance(key, str) else key
            for name, key in (
                ("embeddings", embeddings_key),
                ("is_query", is_query_key),
                ("pid", person_key),
                ("cid", camera_key),
            )
        }
        self._extract_source = {
            "embeddings": "output",
            "is_query": "input",
            "pid": "input",
            "cid": "input",
        }
        self._fields = REID_FIELDS
        self._accumulative_fields = {
            data_split: {field: None for field in self._fields}
            for data_split in DATA_SPLITS
        }
        self._indices = {key: 0 for key in DATA_SPLITS}
        self._sizes = {key: None for key in DATA_SPLITS}

    def on_batch_end(self, runner: "IRunner"):
        """On batch end action"""
        batch_data = self._get_batch_data(runner=runner)

        query_mask = batch_data["is_query"].type(TORCH_BOOL)
        gallery_mask = ~query_mask

        for data_split, mask in (
            ("query", query_mask), ("gallery", gallery_mask),
        ):
            embeddings = batch_data["embeddings"][mask].cpu()
            pids = batch_data["pids"][mask].cpu()
            cids = batch_data["cids"][mask].cpu()

            if self._accumulative_fields[data_split]["embeddings"] is None:
                self._accumulative_fields[data_split]["embeddings"] = torch.empty(
                    size=(self._sizes[data_split], embeddings.shape[1])
                )

            self._accumulate_batch(
                data_split=data_split,
                data={"embeddings": embeddings, "pids": pids, "cids": cids},
            )

    def on_loader_end(self, runner: "IRunner"):
        """On loader end action"""
        self._check_completeness()
        pid_conformity_matrix = self._get_conformity_matrix(
            query_labels=self._accumulative_fields["query"]["pids"],
            gallery_labels=self._accumulative_fields["gallery"]["pids"],
        )
        cid_conformity_matrix = self._get_conformity_matrix(
            query_labels=self._accumulative_fields["query"]["cids"],
            gallery_labels=self._accumulative_fields["gallery"]["cids"],
        )
        # Now we are going to generate a mask that should show if
        # a sample from gallery can be used during model scoring on the query
        # sample.
        # There is only one case when the label shouldn't be used for:
        # if query sample is a photo of the person pid_i taken from camera
        # cam_j and the gallery sample is a photo of the same person pid_i
        # from the same camera cam_j. All another cases are available.
        labels_available = ~(pid_conformity_matrix * cid_conformity_matrix)

        if (labels_available.max(dim=1).values == 0).any():
            ValueError(
                "There is a sample in query that has no relevant samples "
                "in gallery."
            )

        for key in self.list_args:
            metric = self._metric_fn(
                query_embeddings=self._accumulative_fields["query"][
                    "embeddings"
                ],
                gallery_embeddings=self._accumulative_fields["gallery"][
                    "embeddings"
                ],
                conformity_matrix=pid_conformity_matrix,
                mask=labels_available,
                topk=key,
            )
            runner.loader_metrics[f"{self._prefix}{key:02}"] = metric
        self._reset_fields()


__all__ = ["CMCScoreCallback", "ReidCMCScoreCallback"]
