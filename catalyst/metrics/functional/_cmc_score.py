import torch


def cmc_score_count(
    distances: torch.Tensor, conformity_matrix: torch.Tensor, topk: int = 1,
) -> float:
    """
    Function to count CMC from distance matrix and conformity matrix.

    Args:
        distances: distance matrix shape of (n_embeddings_x, n_embeddings_y)
        conformity_matrix: binary matrix with 1 on same label pos and 0 otherwise
        topk: number of top examples for cumulative score counting

    Returns:
        cmc score
    """
    perm_matrix = torch.argsort(distances)
    position_matrix = torch.argsort(perm_matrix)
    conformity_matrix = conformity_matrix.type(torch.bool)

    position_matrix[~conformity_matrix] = topk + 1  # value large enough not to be counted

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
        conformity_matrix: binary matrix with 1 on same label pos and 0 otherwise
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
    available_samples: torch.Tensor,
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
        available_samples: tensor of shape (query_size, gallery_size),
            available_samples[i][j] == 1 means that j-th element of gallery
            should be used while scoring i-th query one
        topk: number of top examples for cumulative score counting

    Returns:
        cmc score with mask

    Raises:
        ValueError: if there are items that have different labels and are
            unavailable for each other according to availability matrix
    """
    if not available_samples[conformity_matrix == 0].all():
        raise ValueError(
            "There is something wrong with available_samples matrix.\n"
            "If we calculate masked_cmc_score for person pid_i, we should "
            "take into account all the photos of people other than pid_i; "
            "it means that all of them should be available according to "
            "available_samples matrix.\n"
            "It seems that it's not so for your one."
        )
    distances = torch.cdist(query_embeddings, gallery_embeddings)
    distances[~available_samples] = float("inf")
    return cmc_score_count(distances, conformity_matrix, topk)


__all__ = ["cmc_score_count", "cmc_score", "masked_cmc_score"]
