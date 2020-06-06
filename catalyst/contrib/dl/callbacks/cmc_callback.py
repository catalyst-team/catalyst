from typing import List, Optional

import torch

from catalyst.dl import Callback


def euclidean_distance(
    x: torch.Tensor, y: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Computes euclidean distance between embeddings
    Args:
        x: matrix with shape of (n_objects, emb_dim)
        y: matrix with shape of (n_objects, emb_dim)

    Returns: matrix shape of (n_objects, n_objects)

    """

    x_norm = (x ** 2).sum(1).view(-1, 1)
    if y is not None:
        y_t = torch.transpose(y, 0, 1)
        y_norm = (y ** 2).sum(1).view(1, -1)
    else:
        y_t = torch.transpose(x, 0, 1)
        y_norm = x_norm.view(1, -1)
    # ||x - y||^2 = ||x||^2 - 2<x,y> + ||y||^2
    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    return torch.relu(dist)


def cmc_score(
    gallery_embeddings: torch.Tensor,
    query_embeddings: torch.Tensor,
    conformity_matrix: torch.Tensor,
    topk: int = 1,
) -> float:
    """

    Args:
        gallery_embeddings:
        query_embeddings:
        conformity_matrix:
        topk:

    Returns:

    """
    distances = euclidean_distance(gallery_embeddings, query_embeddings)
    position_matrix = torch.argsort(distances, dim=1)
    position_matrix[~conformity_matrix] = (
        topk + 1
    )  # value large enough not to be counted
    closest = position_matrix.argmin(dim=1)
    k_mask = (closest < topk).type(torch.float)
    return k_mask.mean().item()


class CMCScoreCallback(Callback):
    """Cumulative Matching Characteristics"""

    def __init__(
        self,
        conformity_matrix: torch.Tensor,
        gallery_key: str = "embeddings_gallery",
        queries_key: str = "embeddings_queries",
        prefix: str = "cmc",
        map_args: List[int] = None,
    ):
        """
        Args:
            output_key (str): embeddings key in output dict
            prefix (str): key for the metric's name
            map_args (List[int]): specifies which map@K to log.
                [1] - map@1
                [1, 3] - map@1 and map@3
                [1, 3, 5] - map@1, map@3 and map@5
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
        self._queries_embeddings = None

    def on_batch_end(self, runner: "IRunner"):
        current_gallery = runner.output.get(self.gallery_key, None)
        current_queries = runner.output.get(self.queries_key, None)
        if current_gallery is not None:
            if self._gallery_embeddings is None:
                self._gallery_embeddings = current_gallery
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
        for k in self.list_args:
            metric = self._metric_fn(
                self._gallery_embeddings,
                self._queries_embeddings,
                self.conformity_matrix,
                k,
            )
            runner.loader_metrics[f"{self._prefix}_{k}"] = metric
