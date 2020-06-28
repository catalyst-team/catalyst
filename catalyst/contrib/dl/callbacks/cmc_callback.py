from typing import List

import torch

from catalyst.dl import Callback


def _cmc_score_count(
    distances: torch.Tensor, conformity_matrix: torch.Tensor, topk: int = 1,
) -> float:
    """
    More convenient to write tests with distance_matrix
    Args:
        distances: distance matrix shape of (n_embeddings_x, n_embeddings_y)
        conformity_matrix: binary matrix with 1 on same label pos
            and 0 otherwise
        topk: number of top examples for cumulative score counting

    Returns:
        cmc score
    """
    position_matrix = torch.argsort(distances, dim=1)
    conformity_matrix = conformity_matrix.type(torch.bool)
    position_matrix[~conformity_matrix] = (
        topk + 1
    )  # value large enough not to be counted

    closest = position_matrix.min(dim=1)[0]
    k_mask = (closest < topk).type(torch.float)
    return k_mask.mean().item()


def cmc_score(
    query_embeddings: torch.Tensor,
    gallery_embeddings: torch.Tensor,
    conformity_matrix: torch.Tensor,
    topk: int = 1,
) -> float:
    """

    Args:
        query_embeddings: tensor shape of (n_embeddings, embedding_dim)
            embeddings of the objects in querry
        gallery_embeddings: tensor shape of (n_embeddings, embedding_dim)
            embeddings of the objects in gallery
        conformity_matrix: binary matrix with 1 on same label pos
            and 0 otherwise
        topk: number of top examples for cumulative score counting

    Returns:
        cmc score
    """
    distances = torch.cdist(query_embeddings, gallery_embeddings)
    return _cmc_score_count(distances, conformity_matrix, topk)


class CMCScoreCallback(Callback):
    """Cumulative Matching Characteristics"""

    def __init__(
        self,
        conformity_matrix: torch.Tensor,
        queries_key: str = "embeddings_queries",
        gallery_key: str = "embeddings_gallery",
        prefix: str = "cmc",
        map_args: List[int] = None,
    ):
        """
        This callback is design to count cumulative matching characteristics.
        Expecting that your dataset has two keys in output dict
        `queries_key` and `gallery_key`. If current object is in
        querries set you should provide `None` value to `gallery_key`
        and embedding of this object should be stored in `queries_key`.
        if object is in gallery you should provide `None` value to
        `queries_key` and embedding of this object should be stored in
        `gallery_key`.
        On batch end callback accumulate
        all embeddings
        Args:
            output_key (str): embeddings key in output dict
            prefix (str): key for the metric's name
            map_args (List[int]): specifies which map@K to log.
                [1] - cmc@1
                [1, 3] - cmc@1 and cmc@3
                [1, 3, 5] - cmc@1, cmc@3 and cmc@5
            num_classes (int): number of classes to calculate ``map_args``
                if ``map_args`` is None
        """
        self.conformity_matrix = conformity_matrix
        self.list_args = map_args
        self._metric_fn = cmc_score
        self._prefix = prefix
        self.gallery_key = gallery_key
        self.queries_key = queries_key
        self._gallery_embeddings: torch.Tensor = None
        self._queries_embeddings: torch.Tensor = None

    def on_batch_end(self, runner: "IRunner"):
        """On batch end action"""
        current_gallery = (
            runner.output.get(self.gallery_key, None).detach().cpu()
        )
        current_queries = (
            runner.output.get(self.queries_key, None).detach().cpu()
        )
        if current_gallery is not None:
            if self._gallery_embeddings is None:
                self._gallery_embeddings = current_gallery.d
            else:
                self._gallery_embeddings = torch.cat(
                    [self._gallery_embeddings, current_gallery], dim=0
                )
        if current_queries is not None:
            if self._queries_embeddings is None:
                self._queries_embeddings = current_queries
            else:
                self._queries_embeddings = torch.cat(
                    [self._queries_embeddings, current_queries], dim=0
                )

    def on_loader_end(self, runner: "IRunner"):
        """On loader end action"""
        for k in self.list_args:
            metric = self._metric_fn(
                self._gallery_embeddings,
                self._queries_embeddings,
                self.conformity_matrix,
                k,
            )
            runner.loader_metrics[f"{self._prefix}_{k}"] = metric
