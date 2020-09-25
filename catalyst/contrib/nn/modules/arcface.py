import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcFace(nn.Module):
    """Implementation of ArcFace loss for metric learning.

    .. _ArcFace: Additive Angular Margin Loss for Deep Face Recognition:
        https://arxiv.org/abs/1801.07698v1

    Example:
        >>> layer = ArcFace(5, 10, s=1.31, m=0.5)
        >>> loss_fn = nn.CrosEntropyLoss()
        >>> embedding = torch.randn(3, 5, requires_grad=True)
        >>> target = torch.empty(3, dtype=torch.long).random_(5)
        >>> output = layer(embedding, target)
        >>> loss = loss_fn(output, target)
        >>> loss.backward()

    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        s: float = 64.0,
        m: float = 0.5,
        eps: float = 1e-6,
    ):
        """
        Args:
            in_features (int): size of each input sample.
            out_features (int): size of each output sample.
            s (float, optional): norm of input feature,
                Default: ``64.0``.
            m (float, optional): margin.
                Default: ``0.5``.
            eps (float, optional): operation accuracy.
                Default: ``1e-6``.
        """
        super(ArcFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.threshold = math.pi - m
        self.eps = eps

        self.weight = nn.Parameter(
            torch.FloatTensor(out_features, in_features)
        )
        nn.init.xavier_uniform_(self.weight)

    def __repr__(self) -> str:
        """Object representation."""
        return (
            "ArcFace("
            + ",".join(
                [
                    f"in_features={self.in_features}",
                    f"out_features={self.out_features}",
                    f"s={self.s}",
                    f"m={self.m}",
                    f"eps={self.eps}",
                ]
            )
            + ")"
        )

    def forward(
        self, input: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            input (torch.Tensor): input features,
                expected shapes ``BxF`` where ``B``
                is batch dimension and ``F`` is an
                input feature dimension.
            target (torch.Tensor): target classes,
                expected shapes ``B`` where
                ``B`` is batch dimension.

        Returns:
            tensor (logits) with shapes ``BxC``
            where ``C`` is a number of classes.
        """
        cos_theta = F.linear(F.normalize(input), F.normalize(self.weight))
        theta = torch.acos(
            torch.clamp(cos_theta, -1.0 + self.eps, 1.0 - self.eps)
        )

        one_hot = torch.zeros_like(cos_theta, device=input.device)
        one_hot.scatter_(1, target.view(-1, 1).long(), 1)

        mask = torch.where(
            theta > self.threshold, torch.zeros_like(one_hot), one_hot
        )

        logits = torch.cos(torch.where(mask.bool(), theta + self.m, theta))
        logits *= self.s

        return logits
