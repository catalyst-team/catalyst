from typing import List, Union

import torch
from torch import nn


class NTXentLoss(nn.Module):
    """
    NTXent loss from `A Simple Framework for Contrastive Learning of Visual Representations`_ paper.

    Adapter from:
    https://arxiv.org/abs/2002.05709

    Example:
        >>> import torch
        >>> from torch.nn import functional as F
        >>> from catalyst.contrib.nn import NTXent
        >>>
        >>> features = F.normalize(torch.rand(256, 64, requires_grad=True))
        >>> labels = torch.randint(high=10, size=(256))
        >>> criterion = NTXent(tau=0.25)
        >>> criterion(features1, features2)
    """

    def __init__(self, tau: float) -> None:
        """

        Args:
            tau: tau to use
        """
        super().__init__()
        self.tau = tau
        self.cosineSim = nn.CosineSimilarity()

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
        bs = features1.shape[0]

        pos_loss = self.cosineSim(features1, features2).sum(dim=0) / self.tau
        list_neg_loss = [
            torch.exp(self.cosineSim(features1, torch.roll(features2, i, 1))) for i in range(1, bs)
        ]
        neg_loss = torch.stack(list_neg_loss, dim=0).sum(dim=0).sum(dim=0)

        loss = pos_loss - torch.log(neg_loss / self.tau)
        return loss
