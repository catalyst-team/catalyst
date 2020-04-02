from typing import List, Union

import torch
from torch import nn

from .functional import margin_loss


class MarginLoss(nn.Module):
    """@TODO: Docs. Contribution is welcome."""

    def __init__(
        self,
        alpha: float = 0.2,
        beta: float = 1.0,
        skip_labels: Union[int, List[int]] = -1,
    ):
        """
        Args:
            alpha (float):
            beta (float):
            skip_labels (int or List[int]):

        @TODO: Docs. Contribution is welcome.
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.skip_labels = skip_labels

    def forward(
        self, embeddings: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Forward propagation method for the margin loss.

        @TODO: Docs. Contribution is welcome.
        """
        return margin_loss(
            embeddings,
            targets,
            alpha=self.alpha,
            beta=self.beta,
            skip_labels=self.skip_labels,
        )
