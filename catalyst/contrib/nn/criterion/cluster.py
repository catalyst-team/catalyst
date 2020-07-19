import torch
from torch import nn

from catalyst.contrib.nn.criterion.functional import euclidean_distance


class ClusterLoss(nn.Module):
    """
    Cluster Loss for Person Re-Identification
    https://arxiv.org/pdf/1812.10325.pdf
    """

    def __init__(self, margin: float = 1) -> None:
        """
        Args:
            margin: margin in cluster loss with hard sampling
            strategy.
        """
        super(ClusterLoss, self).__init__()
        self.margin = margin

    def _get_label_mask(self, labels: torch.Tensor) -> torch.Tensor:
        """
        Generate matrix of bool of shape (n_unique_labels, batch_size),
        where n_unique_labels is a number of unique labels
        in the batch; matrix[i, j] is True if j-th element of
        dataset relates to i-th class and False otherwise.
        Args:
            labels: labels of the batch, shape (batch_size,)
        Returns:
            matrix of indices of classes in batch
        """
        unique_labels = torch.unique(labels)
        label_mask = torch.eq(
            labels.unsqueeze(0), unique_labels.unsqueeze(1)
        ).bool()
        return label_mask

    def _count_intra_class_distances(
        self, embeddings: torch.Tensor, mean_vectors: torch.Tensor
    ) -> torch.Tensor:
        """
        Count matrix of distances from mean vector of each class to it's
        samples embeddings.
        Args:
            embeddings: tensor of shape (p, k, embed_dim) where p is a number
            of classes in the batch, k is a number of samples for each class
            mean_vectors: tensor of shape (p, embed_dim) -- mean vectors
            of each class in the batch
        Returns:
            tensor of shape (p, k) -- matrix of distances from mean vectors to
            related samples in the batch
        """
        p, k, embed_dim = embeddings.shape
        # Create (p, k, embed_dim) tensor of mean vectors for each class
        mean_vectors = mean_vectors.unsqueeze(1).repeat((1, k, 1))
        # Count euclidean distance between embeddings and mean vectors
        distances = torch.pow(embeddings - mean_vectors, 2).sum(2)
        return distances

    def _count_inter_class_distances(
        self, mean_vectors: torch.Tensor
    ) -> torch.Tensor:
        """
        Count matrix of distances from mean vectors of classes to each other
        Args:
            mean_vectors: tensor of shape (p, embed_dim) -- mean vectors
            of classes
        Returns:
            tensor of shape (p, p) -- matrix of distances between mean vectors
        """
        p, _ = mean_vectors.shape
        distance = euclidean_distance(mean_vectors, mean_vectors)
        return distance

    def _skip_diagonal(self, matrix: torch.Tensor) -> torch.Tensor:
        """
        Get all elements from matrix except diagonal ones.
        Args:
            matrix: tensor of shape (p, p)
        Returns:
            modified matrix of shape (p, p - 1) created from matrix by
            deletion of diagonal elements
        """
        p, _ = matrix.shape
        # Create matrix of indices with zero diagonal
        indices = torch.ones(size=(p, p)) - torch.diag(torch.ones(p))
        indices = indices.bool()
        diagonal_free = matrix[indices].view((p, p - 1))
        return diagonal_free

    def _batch_hard_cluster_loss(
        self, embeddings: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Count cluster loss with hard sampling over the batch.
        Each batch should contains k samples for p classes.
        Args:
            embeddings: tensor of shape (batch_size; embed_dim)
            where batch_size = k * p
            labels: labels of the batch, of size (batch_size,)

        Returns:
            scalar tensor containing the cluster loss
        """
        # Get matrix of indices of labels in batch
        labels_mask = self._get_label_mask(labels)
        p = labels_mask.shape[0]
        k = labels.shape[0] // p

        # Validate batch: expected to get batch
        # with k samples for p classes
        assert (labels_mask.sum(1, keepdim=True) == k).all(), ValueError(
            f"For batch of shape {labels_mask.shape} with "
            f"{p} classes required {k} samples for ech class."
        )

        embed_dim = embeddings.shape[-1]
        # Reshape embeddings to groups of (p, k, embed_dim) ones,
        # each i-th group contains embeddings of i-th class.
        embeddings = embeddings.repeat((p, 1, 1))
        embeddings = embeddings[labels_mask].view((p, -1, embed_dim))

        # Count mean vectors for each class in batch
        mean_vectors = embeddings.mean(1)

        d_intra = self._count_intra_class_distances(embeddings, mean_vectors)
        # Count the distances to the sample farthest from mean vector
        # for each class.
        d_intra = d_intra.max(1).values
        # Count matrix of distances from mean vectors to each other
        d_inter = self._count_inter_class_distances(mean_vectors)
        # For each class mean vector get the closest mean vector
        d_inter = self._skip_diagonal(d_inter)
        d_inter = d_inter.min(1).values

        # Count batch loss. For each class i:
        # loss_i = max(d_intra - d_inter + alpha, 0)
        # loss = loss_1 + loss_2 + ... + loss_p
        loss = torch.mean(torch.relu(d_intra - d_inter + self.margin))
        return loss

    def forward(
        self, embeddings: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward propagation method for cluster loss.
        Args:
            embeddings: tensor of shape (batch_size, embed_dim)
            targets: labels of the batch, shape (batch_size,)
        Returns:
            cluster loss for the batch
        """
        return self._batch_hard_cluster_loss(embeddings, targets)
