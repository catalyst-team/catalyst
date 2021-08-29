from math import e

import torch
from torch import nn


class NTXentLoss(nn.Module):
    """A Contrastive embedding loss.

    It has been proposed in `A Simple Framework
    for Contrastive Learning of Visual Representations`_.

    Example:

    .. code-block:: python

        import torch
        from torch.nn import functional as F
        from catalyst.contrib.nn import NTXentLoss

        embeddings_left = F.normalize(torch.rand(256, 64, requires_grad=True))
        embeddings_right = F.normalize(torch.rand(256, 64, requires_grad=True))
        criterion = NTXentLoss(tau = 0.1)
        criterion(embeddings_left, embeddings_right)

    .. _`A Simple Framework for Contrastive Learning of Visual Representations`:
        https://arxiv.org/abs/2002.05709
    """

    def __init__(self, tau: float, reduction: str = "mean") -> None:
        """

        Args:
            tau: temperature
            reduction (string, optional): specifies the reduction to apply to the output:
                ``"none"`` | ``"mean"`` | ``"sum"``.
                ``"none"``: no reduction will be applied,
                ``"mean"``: the sum of the output will be divided by the number of
                positive pairs in the output,
                ``"sum"``: the output will be summed.

        Raises:
            ValueError: if reduction is not mean, sum or none
        """
        super().__init__()
        self.tau = tau
        self.cosine_sim = nn.CosineSimilarity()
        self.reduction = reduction

        if self.reduction not in ["none", "mean", "sum"]:
            raise ValueError(f"Reduction should be: mean, sum, none. But got - {self.reduction}!")

    def forward(self, features1: torch.Tensor, features2: torch.Tensor) -> torch.Tensor:
        """

        Args:
            features1: batch with samples features of shape
                [bs; feature_len]
            features2: batch with samples features of shape
                [bs; feature_len]

        Returns:
            torch.Tensor: NTXent loss
        """
        assert (
            features1.shape == features2.shape
        ), f"Invalid shape of input features: {features1.shape} and {features2.shape}"

        feature_matrix = torch.cat([features1, features2])
        feature_matrix = torch.nn.functional.normalize(feature_matrix)
        # if ||x|| = ||y|| = 1 then||x-y||^2 = 2 - 2<x,y>
        cosine_matrix = (2 - torch.cdist(feature_matrix, feature_matrix) ** 2) / 2

        # todo try different places for temparature
        exp_cosine_matrix = torch.exp(cosine_matrix / self.tau)
        # neg part of the loss
        # torch.exp(1) self similarity
        exp_sim_sum = exp_cosine_matrix.sum(dim=1) - e ** (1 / self.tau)
        neg_loss = torch.log(exp_sim_sum)
        pos_loss = self.cosine_sim(features1, features2) / self.tau
        pos_loss = torch.cat([pos_loss, pos_loss])

        # 2*poss_loss (i,j) and (j,i)
        loss = -pos_loss + neg_loss
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        return loss
