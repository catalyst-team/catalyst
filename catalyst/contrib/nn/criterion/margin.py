from typing import List, Union

import torch.nn as nn

from .functional import margin_loss


class MarginLoss(nn.Module):
    def __init__(
        self,
        alpha: float = 0.2,
        beta: float = 1.0,
        skip_labels: Union[int, List[int]] = -1,
    ):
        """
        Constructor method for the MarginLoss class.
        Args:
            alpha:
            beta:
            skip_labels:
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.skip_labels = skip_labels

    def forward(self, embeddings, targets):
        return margin_loss(
            embeddings,
            targets,
            alpha=self.alpha,
            beta=self.beta,
            skip_labels=self.skip_labels,
        )
