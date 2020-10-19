import torch

TORCH_BOOL = torch.bool if torch.__version__ > "1.1.0" else torch.ByteTensor


def cmc_score_count(
    distances: torch.Tensor, conformity_matrix: torch.Tensor, topk: int = 1,
) -> float:
    """
    Function to count CMC from distance matrix and conformity matrix.

    Args:
        distances: distance matrix shape of (n_embeddings_x, n_embeddings_y)
        conformity_matrix: binary matrix with 1 on same label pos
            and 0 otherwise
        topk: number of top examples for cumulative score counting

    Returns:
        cmc score
    """
    perm_matrix = torch.argsort(distances)
    position_matrix = torch.argsort(perm_matrix)
    conformity_matrix = conformity_matrix.type(TORCH_BOOL)

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
    Function to count CMC score from query and gallery embeddings.

    Args:
        query_embeddings: tensor shape of (n_embeddings, embedding_dim)
            embeddings of the objects in query
        gallery_embeddings: tensor shape of (n_embeddings, embedding_dim)
            embeddings of the objects in gallery
        conformity_matrix: binary matrix with 1 on same label pos
            and 0 otherwise
        topk: number of top examples for cumulative score counting

    Returns:
        cmc score
    """
    distances = torch.cdist(query_embeddings, gallery_embeddings)
    return cmc_score_count(distances, conformity_matrix, topk)


def masked_cmc_score(
    query_embeddings: torch.Tensor,
    gallery_embeddings: torch.Tensor,
    conformity_matrix: torch.Tensor,
    mask: torch.Tensor,
    topk: int = 1,
) -> float:
    """

    Args:
        query_embeddings: tensor shape of (n_embeddings, embedding_dim)
            embeddings of the objects in query
        gallery_embeddings: tensor shape of (n_embeddings, embedding_dim)
            embeddings of the objects in gallery
        conformity_matrix: binary matrix with 1 on same label pos
            and 0 otherwise
        mask: tensor of shape (query_size, gallery_size), mask[i][j] == 1
            means that j-th element of gallery should be used while scoring
            i-th query one
        topk: number of top examples for cumulative score counting

    Returns:
        cmc score with mask
    """
    query_size = query_embeddings.shape[0]
    score = torch.empty(size=(query_size,))
    for i in range(query_size):
        _query_embeddings = query_embeddings[i].reshape(1, -1)
        _gallery_embeddings = gallery_embeddings[mask[i]]
        _conformity_matrix = conformity_matrix[i, mask[i]].reshape(1, -1)
        score[i] = cmc_score(
            query_embeddings=_query_embeddings,
            gallery_embeddings=_gallery_embeddings,
            conformity_matrix=_conformity_matrix,
            topk=topk,
        )
    return score.mean().item()


__all__ = ["cmc_score_count", "cmc_score", "masked_cmc_score"]
