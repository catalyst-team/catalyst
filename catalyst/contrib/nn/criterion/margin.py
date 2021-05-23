from typing import List, Union

import torch
from torch import nn

from catalyst.contrib.nn.criterion.functional import margin_loss


class MarginLoss(nn.Module):
    """Margin loss criterion"""

    def __init__(
        self, alpha: float = 0.2, beta: float = 1.0, skip_labels: Union[int, List[int]] = -1,
    ):
        """
        Margin loss constructor.

        Args:
            alpha: alpha
            beta: beta
            skip_labels (int or List[int]): labels to skip
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.skip_labels = skip_labels

    def forward(self, embeddings: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward method for the margin loss.

        Args:
            embeddings: tensor with embeddings
            targets: tensor with target labels

        Returns:
            computed loss
        """
        return margin_loss(
            embeddings, targets, alpha=self.alpha, beta=self.beta, skip_labels=self.skip_labels,
        )


__all__ = ["MarginLoss"]
