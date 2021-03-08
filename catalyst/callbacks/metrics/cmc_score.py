from typing import List

from catalyst.callbacks.metric import LoaderMetricCallback
from catalyst.metrics._cmc_score import CMCMetric
from catalyst.metrics.functional import get_default_topk_args
from catalyst.utils.distributed import get_rank


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
        embeddings_key: str,
        labels_key: str,
        is_query_key: str,
        topk_args: List[int] = None,
        num_classes: int = None,
        prefix: str = None,
        suffix: str = None,
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
        self.list_args = topk_args or get_default_topk_args(num_classes)
        super().__init__(
            metric=CMCMetric(
                embeddings_key=embeddings_key,
                labels_key=labels_key,
                is_query_key=is_query_key,
                prefix=prefix,
                suffix=suffix,
            ),
            input_key=[embeddings_key, is_query_key],
            target_key=[labels_key],
        )
        assert get_rank() < 0, "No DDP support implemented yet"


__all__ = ["CMCScoreCallback"]
