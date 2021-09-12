from math import e, log

import torch
from torch import nn


class SupervisedContrastiveLoss(nn.Module):
    """A Contrastive embedding loss that uses targets.

    It has been proposed in `Supervised Contrastive Learning`_.

    .. _`Supervised Contrastive Learning`:
        https://arxiv.org/pdf/2004.11362.pdf
    """

    def __init__(self, tau: float, reduction: str = "mean", pos_aggregation="in") -> None:
        """
        Args:
            tau: temperature
            reduction (string, optional): specifies the reduction to apply to the output:
                ``"none"`` | ``"mean"`` | ``"sum"``.
                ``"none"``: no reduction will be applied,
                ``"mean"``: the sum of the output will be divided by the number of
                positive pairs in the output,
                ``"sum"``: the output will be summed.
            pos_aggregation (string, optional): specifies the place of positive pairs aggregation:
                ``"in"`` | ``"out"``.
                ``"in"``: maximization of log(average positive exponentiate similarity)
                ``"out"``: maximization of average positive similarity
        Raises:
            ValueError: if reduction is not mean, sum or none
        """
        super().__init__()
        self.tau = tau
        self.reduction = reduction
        self.positive_aggregation = pos_aggregation

        if self.reduction not in ["none", "mean", "sum"]:
            raise ValueError(f"Reduction should be: mean, sum, none. But got - {self.reduction}!")
        if self.positive_aggregation in ["in", "out"]:
            raise ValueError(
                f"Positive aggregation should be: in or out. But got - {self.pos_aggregation}!"
            )

    def forward(self, features: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [bs; feature_len]
            targets: [bs]

        Returns:
            computed loss
        """
        # torch.exp(1/tau) self similarity
        self_similarity = 1 / self.tau
        exp_self_similarity = e ** (1 / self.tau)
        # if ||x|| = ||y|| = 1 then||x-y||^2 = 2 - 2<x,y>
        cosine_matrix = (2 - torch.cdist(features, features) ** 2) / 2
        exp_cosine_matrix = torch.exp(cosine_matrix / self.tau)
        # positive part of the loss
        pos_place = targets.repeat(targets.shape[0], 1) == targets.reshape(targets.shape[0], 1)
        # aggregation of postive pairs

        number_of_positives = pos_place.sum(dim=1) - 1
        if self.pos_aggregation == "in":
            pos_loss = ((exp_cosine_matrix * pos_place).sum(dim=1) - self_similarity) - log(
                number_of_positives
            )
        elif self.pos_aggregation == "out":
            pos_loss = (
                (torch.log(exp_cosine_matrix) * pos_place).sum(dim=1) - 1 / self.tau
            ) / number_of_positives

        # neg part of the loss
        exp_sim_sum = exp_cosine_matrix.sum(dim=1) - exp_self_similarity
        neg_loss = torch.log(exp_sim_sum)

        # 2*poss_loss (i,j) and (j,i)
        loss = -pos_loss + neg_loss
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        return loss
