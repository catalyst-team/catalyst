from typing import Dict, List, Optional, Union

import torch

from catalyst.core import Callback, CallbackOrder
from catalyst.data import QueryGalleryDataset
from catalyst.metrics.cmc_score import cmc_score, masked_cmc_score
from catalyst.metrics.functional import get_default_topk_args

TAccumulative = Union[List[int], torch.Tensor]


class AccumulatorCallback(Callback):
    """
    This callback can accumulate batch data throughout epoch.
    It can accumulate primitives and 1D arrays.
    """

    def __init__(
        self, input_keys: List[str], output_keys: List[str],
    ):
        """
        Args:
            input_keys: runner's input keys that should be accumulated
                in callback
            output_keys: runner's output keys that should be accumulated
                in callback
        """
        super().__init__(order=CallbackOrder.Metric)

        assert not set(input_keys).intersection(set(output_keys)), ValueError(
            "Input and output keys should be different."
        )

        self._input_keys = input_keys
        self._output_keys = output_keys

        self._cur_idx = 0
        self._storage_size: Optional[int] = None
        self._storage: Optional[Dict[str, TAccumulative]] = None

    def _check_completeness(self):
        """Check if we have accumulated all the samples for dataset."""
        assert self._storage_size == self._cur_idx, (
            f"An error occurred during the accumulation process: expected "
            f"to get {self._storage_size} elements, got {self._cur_idx}."
        )

    def on_loader_start(self, runner: "IRunner") -> None:
        """On loader start action"""
        dataset = runner.loaders[runner.loader_name].dataset
        self._storage = {}
        self._storage_size = len(dataset)
        for key in self._input_keys + self._output_keys:
            self._storage[key] = None

    def _reset_fields(self) -> None:
        """Reset all the accumulative fields"""
        self._cur_idx = 0
        self._storage_size = None
        self._storage = None

    def on_loader_end(self, runner: "IRunner") -> None:
        """On loader end action"""
        self._check_completeness()
        self._reset_fields()

    def on_batch_end(self, runner: "IRunner") -> None:
        """On batch end action"""
        batch_size = None
        for keys, source in [
            (self._input_keys, runner.input),
            (self._output_keys, runner.output),
        ]:
            for key in keys:
                source_shape = source[key].shape
                if batch_size is not None and batch_size != source_shape[0]:
                    raise ValueError(
                        "Different fields of input and output dicts contain "
                        "different number of items."
                    )
                batch_size = source_shape[0]

                is_first_step = self._storage[key] is None
                if is_first_step:
                    field_shape = (
                        (self._storage_size, source_shape[1])
                        if len(source_shape) > 1
                        else (self._storage_size,)
                    )
                    self._storage[key] = torch.empty(
                        size=field_shape, dtype=source[key].dtype
                    )

                indices_range = self._cur_idx + torch.arange(batch_size)
                self._storage[key][indices_range] = source[key]
        self._cur_idx += batch_size


class CMCScoreCallback(AccumulatorCallback):
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
        topk_args: Optional[List[int]] = None,
        num_classes: Optional[int] = None,
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
            embeddings_key: key of embeddings in runner's output dict
            labels_key: key of labels in runner's input dict
            is_query_key: key of is_query field in runner's input dict
            prefix: key for the metric's name
            topk_args: specifies which cmc@K to log.
                [1] - cmc@1
                [1, 3] - cmc@1 and cmc@3
                [1, 3, 5] - cmc@1, cmc@3 and cmc@5
            num_classes: number of classes to calculate ``accuracy_args``
                if ``topk_args`` is None
        """
        super().__init__(
            input_keys=[labels_key, is_query_key],
            output_keys=[embeddings_key],
        )

        self._embedding_key = embeddings_key
        self._labels_key = labels_key
        self._is_query_key = is_query_key

        self._prefix = prefix
        self.list_args = topk_args or get_default_topk_args(num_classes)

        self._metric_fn = cmc_score

    def on_loader_start(self, runner: "IRunner") -> None:
        """On loader start action"""
        dataset = runner.loaders[runner.loader_name].dataset
        if not isinstance(dataset, QueryGalleryDataset):
            raise ValueError(
                "CMCScoreCallback should be used with QueryGalleryDataset."
            )
        super(CMCScoreCallback, self).on_loader_start(runner=runner)

    def on_loader_end(self, runner: "IRunner") -> None:
        """On loader end action"""
        self._check_completeness()

        query_mask = self._storage[self._is_query_key].bool()
        gallery_mask = ~query_mask

        gallery_labels = self._storage[self._labels_key][gallery_mask]
        query_labels = self._storage[self._labels_key][query_mask]
        query_embeddings = self._storage[self._embedding_key][query_mask]
        gallery_embeddings = self._storage[self._embedding_key][gallery_mask]

        conformity_matrix = gallery_labels == query_labels.reshape(-1, 1)

        for key in self.list_args:
            metric = self._metric_fn(
                query_embeddings=query_embeddings,
                gallery_embeddings=gallery_embeddings,
                conformity_matrix=conformity_matrix,
                topk=key,
            )
            runner.loader_metrics[f"{self._prefix}{key:02}"] = metric
        self._reset_fields()


class ReidCMCScoreCallback(AccumulatorCallback):
    def __init__(
        self,
        embeddings_key: str = "logits",
        pids_key: str = "pid",
        cids_key: str = "cid",
        is_query_key: str = "is_query",
        prefix: str = "cmc",
        topk_args: List[int] = None,
        num_classes: int = None,
    ):
        """
        This callback was designed to count
        cumulative matching characteristics in reid case.
        It counts cmc score with pids and cids logic of reid datasets.

        Args:
            embeddings_key: key of embeddings in runner's output dict
            pids_key: key of pids in runner's input dict
            cids_key: key of cids in runner's input dict
            is_query_key: key of is_query field in runner's input dict
            prefix: key for the metric's name
            topk_args: specifies which cmc@K to log.
                [1] - cmc@1
                [1, 3] - cmc@1 and cmc@3
                [1, 3, 5] - cmc@1, cmc@3 and cmc@5
            num_classes: number of classes to calculate ``accuracy_args``
                if ``topk_args`` is None
        """
        super().__init__(
            input_keys=[pids_key, cids_key, is_query_key],
            output_keys=[embeddings_key],
        )

        self._embedding_key = embeddings_key
        self._pids_key = pids_key
        self._cids_key = cids_key
        self._is_query_key = is_query_key

        self._prefix = prefix
        self.list_args = topk_args or get_default_topk_args(num_classes)

        self._metric_fn = masked_cmc_score

    def on_loader_start(self, runner: "IRunner") -> None:
        """On loader start action"""
        dataset = runner.loaders[runner.loader_name].dataset
        if not isinstance(dataset, QueryGalleryDataset):
            raise ValueError(
                "ReidCMCScoreCallback should be used with "
                "QueryGalleryDataset."
            )
        super().on_loader_start(runner=runner)

    def on_loader_end(self, runner: "IRunner") -> None:
        """On loader end action"""
        self._check_completeness()

        query_mask = self._storage[self._is_query_key].bool()
        gallery_mask = ~query_mask

        gallery_pids = self._storage[self._pids_key][gallery_mask]
        query_pids = self._storage[self._pids_key][query_mask]
        gallery_cids = self._storage[self._cids_key][gallery_mask]
        query_cids = self._storage[self._cids_key][query_mask]

        pid_conformity_matrix = gallery_pids == query_pids.reshape(-1, 1)
        cid_conformity_matrix = gallery_cids == query_cids.reshape(-1, 1)
        # Now we are going to generate a mask that should show if
        # a sample from gallery can be used during model scoring on the query
        # sample.
        # There is only one case when the label shouldn't be used for:
        # if query sample is a photo of the person pid_i taken from camera
        # cam_j and the gallery sample is a photo of the same person pid_i
        # from the same camera cam_j. All other cases are available.
        available_samples = ~(
            pid_conformity_matrix * cid_conformity_matrix
        ).bool()

        if (available_samples.max(dim=1).values == 0).any():
            ValueError(
                "There is a sample in query that has no relevant samples "
                "in gallery."
            )

        for key in self.list_args:
            metric = self._metric_fn(
                query_embeddings=self._storage[self._embedding_key][
                    query_mask
                ],
                gallery_embeddings=self._storage[self._embedding_key][
                    gallery_mask
                ],
                conformity_matrix=pid_conformity_matrix,
                available_samples=available_samples,
                topk=key,
            )
            runner.loader_metrics[f"{self._prefix}{key:02}"] = metric
        self._reset_fields()


__all__ = ["AccumulatorCallback", "CMCScoreCallback", "ReidCMCScoreCallback"]
