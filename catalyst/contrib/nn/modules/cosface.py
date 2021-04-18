import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CosFace(nn.Module):
    """Implementation of
    `CosFace\: Large Margin Cosine Loss for Deep Face Recognition`_.

    .. _CosFace\: Large Margin Cosine Loss for Deep Face Recognition:
        https://arxiv.org/abs/1801.09414

    Args:
        in_features: size of each input sample.
        out_features: size of each output sample.
        s: norm of input feature.
            Default: ``64.0``.
        m: margin.
            Default: ``0.35``.

    Shape:
        - Input: :math:`(batch, H_{in})` where
          :math:`H_{in} = in\_features`.
        - Output: :math:`(batch, H_{out})` where
          :math:`H_{out} = out\_features`.

    Example:
        >>> layer = CosFaceLoss(5, 10, s=1.31, m=0.1)
        >>> loss_fn = nn.CrosEntropyLoss()
        >>> embedding = torch.randn(3, 5, requires_grad=True)
        >>> target = torch.empty(3, dtype=torch.long).random_(10)
        >>> output = layer(embedding, target)
        >>> loss = loss_fn(output, target)
        >>> loss.backward()

    """

    def __init__(  # noqa: D107
        self, in_features: int, out_features: int, s: float = 64.0, m: float = 0.35,
    ):
        super(CosFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m

        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def __repr__(self) -> str:
        """Object representation."""
        rep = (
            "CosFace("
            f"in_features={self.in_features},"
            f"out_features={self.out_features},"
            f"s={self.s},"
            f"m={self.m}"
            ")"
        )
        return rep

    def forward(self, input: torch.Tensor, target: torch.LongTensor = None) -> torch.Tensor:
        """
        Args:
            input: input features,
                expected shapes ``BxF`` where ``B``
                is batch dimension and ``F`` is an
                input feature dimension.
            target: target classes,
                expected shapes ``B`` where
                ``B`` is batch dimension.
                If `None` then will be returned
                projection on centroids.
                Default is `None`.

        Returns:
            tensor (logits) with shapes ``BxC``
            where ``C`` is a number of classes
            (out_features).
        """
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        phi = cosine - self.m

        if target is None:
            return cosine

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, target.view(-1, 1).long(), 1)

        logits = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        logits *= self.s

        return logits


class AdaCos(nn.Module):
    """Implementation of
    `AdaCos\: Adaptively Scaling Cosine Logits for Effectively Learning Deep Face Representations`_.

    .. _AdaCos\: Adaptively Scaling Cosine Logits for\
        Effectively Learning Deep Face Representations:
        https://arxiv.org/abs/1905.00292

    Args:
        in_features: size of each input sample.
        out_features: size of each output sample.
        dynamical_s: option to use dynamical scale parameter.
            If ``False`` then will be used initial scale.
            Default: ``True``.
        eps: operation accuracy.
            Default: ``1e-6``.

    Shape:
        - Input: :math:`(batch, H_{in})` where
          :math:`H_{in} = in\_features`.
        - Output: :math:`(batch, H_{out})` where
          :math:`H_{out} = out\_features`.

    Example:
        >>> layer = AdaCos(5, 10)
        >>> loss_fn = nn.CrosEntropyLoss()
        >>> embedding = torch.randn(3, 5, requires_grad=True)
        >>> target = torch.empty(3, dtype=torch.long).random_(10)
        >>> output = layer(embedding, target)
        >>> loss = loss_fn(output, target)
        >>> loss.backward()

    """  # noqa: E501,W505

    def __init__(  # noqa: D107
        self, in_features: int, out_features: int, dynamical_s: bool = True, eps: float = 1e-6,
    ):
        super(AdaCos, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = math.sqrt(2) * math.log(out_features - 1)
        self.eps = eps

        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def __repr__(self) -> str:
        """Object representation."""
        rep = (
            "AdaCos("
            f"in_features={self.in_features},"
            f"out_features={self.out_features},"
            f"s={self.s},"
            f"eps={self.eps}"
            ")"
        )
        return rep

    def forward(self, input: torch.Tensor, target: torch.LongTensor = None) -> torch.Tensor:
        """
        Args:
            input: input features,
                expected shapes ``BxF`` where ``B``
                is batch dimension and ``F`` is an
                input feature dimension.
            target: target classes,
                expected shapes ``B`` where
                ``B`` is batch dimension.
                If `None` then will be returned
                projection on centroids.
                Default is `None`.

        Returns:
            tensor (logits) with shapes ``BxC``
            where ``C`` is a number of classes
            (out_features).
        """
        cos_theta = F.linear(F.normalize(input), F.normalize(self.weight))
        theta = torch.acos(torch.clamp(cos_theta, -1.0 + self.eps, 1.0 - self.eps))

        if target is None:
            return cos_theta

        one_hot = torch.zeros_like(cos_theta)
        one_hot.scatter_(1, target.view(-1, 1).long(), 1)

        if self.train:
            with torch.no_grad():
                b_avg = (
                    torch.where(
                        one_hot < 1, torch.exp(self.s * cos_theta), torch.zeros_like(cos_theta),
                    )
                    .sum(1)
                    .mean()
                )
                theta_median = theta[one_hot > 0].median()
                theta_median = torch.min(torch.full_like(theta_median, math.pi / 4), theta_median)
                self.s = (torch.log(b_avg) / torch.cos(theta_median)).item()

        logits = self.s * cos_theta
        return logits


__all__ = ["CosFace", "AdaCos"]
