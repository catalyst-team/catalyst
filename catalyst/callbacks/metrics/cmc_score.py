from typing import Dict, Iterable, List, TYPE_CHECKING, Union
from collections import defaultdict

import torch

from catalyst.callbacks.metric import LoaderMetricCallback
from catalyst.data import QueryGalleryDataset
from catalyst.metrics.cmc_score import cmc_score, masked_cmc_score

if TYPE_CHECKING:
    from catalyst.core.runner import IRunner

TORCH_BOOL = torch.bool if torch.__version__ > "1.1.0" else torch.ByteTensor
TAccumulative = Union[List[int], torch.Tensor]


class CMCScoreCallback(LoaderMetricCallback):
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
        topk_args: Iterable[int] = (1,),
    ):
        """
        This callback was designed to count
        cumulative matching characteristics.
        If current object is from query your dataset
        should output `True` in `is_query_key`
        and false if current object is from gallery.
        You can see `QueryGalleryDataset` in
        `catalyst.contrib.datasets.metric_learning` for more information.

        Args:
            embeddings_key: key of embeddings in runner's output dict
            labels_key: key of labels in runner's input dict
            is_query_key: key of is_query field in runner's input dict
            prefix: key for the metric's name
            topk_args: specifies which cmc@K to log.
                [1] - cmc@1
                [1, 3] - cmc@1 and cmc@3
                [1, 3, 5] - cmc@1, cmc@3 and cmc@5
        """
        super().__init__(
            input_key=[labels_key, is_query_key],
            output_key=[embeddings_key],
            metric_fn=cmc_score,
            prefix=prefix,
        )
        self.embedding_key = embeddings_key
        self.label_key = labels_key
        self.is_query_key = is_query_key
        self.topk_args = topk_args

    def _compute_metric_key_value(
        self, output: Dict, input: Dict
    ) -> Dict[str, float]:
        """
        Prepare outputs and compute CMC score for valid loader.

        Args:
            output: data accumulated from runner's output dict
            input: data accumulated from runner's input dict

        Returns:
            dict of CMC scores
        """
        is_query = input[self.is_query_key]
        is_query = is_query.type(TORCH_BOOL)

        query_labels = input[self.label_key][is_query]
        gallery_labels = input[self.label_key][~is_query]

        query_embeddings = output[self.embedding_key][is_query]
        gallery_embeddings = output[self.embedding_key][~is_query]

        conformity_matrix = (
            gallery_labels == query_labels.reshape(-1, 1)
        ).type(TORCH_BOOL)

        metric = defaultdict(float)
        for k in self.topk_args:
            metric[f"{k:02}"] = self.metric(
                query_embeddings=query_embeddings,
                gallery_embeddings=gallery_embeddings,
                conformity_matrix=conformity_matrix,
                topk=k,
            )
        return metric

    def on_loader_start(self, runner: "IRunner"):
        """
        On loader start action.

        Args:
            runner: experiment runner

        Raises:
            ValueError: if dataset that contains accumulative data
                is not a QueryGalleryDataset
        """
        dataset = runner.loaders[runner.loader_name].dataset
        if not isinstance(dataset, QueryGalleryDataset):
            raise ValueError(
                "CMCScoreCallback should be used with QueryGalleryDataset."
            )
        super().on_loader_start(runner=runner)


class ReidCMCScoreCallback(LoaderMetricCallback):
    """Cumulative Matching Characteristics callback."""

    def __init__(
        self,
        embeddings_key: str = "logits",
        pids_key: str = "pid",
        cids_key: str = "cid",
        is_query_key: str = "is_query",
        prefix: str = "cmc",
        topk_args: Iterable[int] = (1,),
    ):
        """
        Init callback params.

        Args:
            embeddings_key: key of embeddings in runner's output dict
            pids_key: key of person id in runner's output dict
            cids_key: key of camera id in runner's output dict
            is_query_key: key of is_query field in runner's input dict
            prefix: key for the metric's name
            topk_args: specifies which cmc@K to log.
                [1] - cmc@1
                [1, 3] - cmc@1 and cmc@3
                [1, 3, 5] - cmc@1, cmc@3 and cmc@5
        """
        super().__init__(
            input_key=[pids_key, cids_key, is_query_key],
            output_key=[embeddings_key],
            metric_fn=masked_cmc_score,
            prefix=prefix,
        )
        self.embeddings_key = embeddings_key
        self.pids_key = pids_key
        self.cids_key = cids_key
        self.is_query_key = is_query_key
        self.topk_args = topk_args

    def _compute_metric_key_value(
        self, output: Dict, input: Dict
    ) -> Dict[str, float]:
        """
        Prepare outputs and compute CMC score for valid loader.

        Args:
            output: data accumulated from runner's output dict
            input: data accumulated from runner's input dict

        Returns:
            dict of CMC scores
        """
        is_query = input[self.is_query_key]
        is_query = is_query.type(TORCH_BOOL)

        query_pids = input[self.pids_key][is_query]
        gallery_pids = input[self.pids_key][~is_query]

        query_cids = input[self.cids_key][is_query]
        gallery_cids = input[self.cids_key][~is_query]

        query_embeddings = output[self.embeddings_key][is_query]
        gallery_embeddings = output[self.embeddings_key][~is_query]

        pid_conformity_matrix = (
            gallery_pids == query_pids.reshape(-1, 1)
        ).type(TORCH_BOOL)
        cid_conformity_matrix = (
            gallery_cids == query_cids.reshape(-1, 1)
        ).type(TORCH_BOOL)

        # Now we are going to generate a mask that should show if
        # a sample from gallery can be used during model scoring on the query
        # sample.
        # There is only one case when the label shouldn't be used for:
        # if query sample is a photo of the person pid_i taken from camera
        # cam_j and the gallery sample is a photo of the same person pid_i
        # from the same camera cam_j. All other cases are available.
        available_samples = ~(
            pid_conformity_matrix * cid_conformity_matrix
        ).type(TORCH_BOOL)

        if (available_samples.max(dim=1).values == 0).any():
            raise ValueError(
                "There is a sample in query that has no relevant samples "
                "in gallery."
            )

        metric = defaultdict(float)
        for k in self.topk_args:
            metric[f"{k:02}"] = self.metric(
                query_embeddings=query_embeddings,
                gallery_embeddings=gallery_embeddings,
                conformity_matrix=pid_conformity_matrix,
                available_samples=available_samples,
                topk=k,
            )
        return metric

    def on_loader_start(self, runner: "IRunner"):
        """
        On loader start action.

        Args:
            runner: experiment runner

        Raises:
            ValueError: if dataset that contains accumulative data
                is not a QueryGalleryDataset
        """
        dataset = runner.loaders[runner.loader_name].dataset
        if not isinstance(dataset, QueryGalleryDataset):
            raise ValueError(
                "ReidCMCScoreCallback should be used with QueryGalleryDataset."
            )
        super().on_loader_start(runner=runner)


__all__ = ["CMCScoreCallback", "ReidCMCScoreCallback"]
