from typing import Tuple

import torch
from torch import nn, Tensor


def _convert_label_to_similarity(normed_features: Tensor, labels: Tensor) -> Tuple[Tensor, Tensor]:
    similarity_matrix = normed_features @ normed_features.transpose(1, 0)
    label_matrix = labels.unsqueeze(1) == labels.unsqueeze(0)

    positive_matrix = label_matrix.triu(diagonal=1)
    negative_matrix = label_matrix.logical_not().triu(diagonal=1)

    similarity_matrix = similarity_matrix.view(-1)
    positive_matrix = positive_matrix.view(-1)
    negative_matrix = negative_matrix.view(-1)
    sp, sn = (
        similarity_matrix[positive_matrix],
        similarity_matrix[negative_matrix],
    )
    return sp, sn


class CircleLoss(nn.Module):
    """
    CircleLoss from `Circle Loss: A Unified Perspective of Pair Similarity Optimization`_ paper.

    Adapter from:
    https://github.com/TinyZeaMays/CircleLoss

    Example:
        >>> import torch
        >>> from torch.nn import functional as F
        >>> from catalyst.contrib.nn import CircleLoss
        >>>
        >>> features = F.normalize(torch.rand(256, 64, requires_grad=True))
        >>> labels = torch.randint(high=10, size=(256))
        >>> criterion = CircleLoss(margin=0.25, gamma=256)
        >>> criterion(features, labels)

    .. _`Circle Loss: A Unified Perspective of Pair Similarity Optimization`:
        https://arxiv.org/abs/2002.10857
    """

    def __init__(self, margin: float, gamma: float) -> None:
        """

        Args:
            margin: margin to use
            gamma: gamma to use
        """
        super().__init__()
        self.margin = margin
        self.gamma = gamma
        self.soft_plus = nn.Softplus()

    def forward(self, normed_features: Tensor, labels: Tensor) -> Tensor:
        """

        Args:
            normed_features: batch with samples features of shape
                [bs; feature_len]
            labels: batch with samples correct labels of shape [bs; ]

        Returns:
            torch.Tensor: circle loss
        """
        sp, sn = _convert_label_to_similarity(normed_features, labels)

        ap = torch.clamp_min(-sp.detach() + 1 + self.margin, min=0.0)
        an = torch.clamp_min(sn.detach() + self.margin, min=0.0)

        delta_p = 1 - self.margin
        delta_n = self.margin

        logit_p = -ap * (sp - delta_p) * self.gamma
        logit_n = an * (sn - delta_n) * self.gamma

        loss = self.soft_plus(torch.logsumexp(logit_n, dim=0) + torch.logsumexp(logit_p, dim=0))

        return loss


__all__ = ["CircleLoss"]
