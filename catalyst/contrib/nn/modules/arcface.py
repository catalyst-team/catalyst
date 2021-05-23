import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcFace(nn.Module):
    """Implementation of
    `ArcFace: Additive Angular Margin Loss for Deep Face Recognition`_.

    .. _ArcFace\: Additive Angular Margin Loss for Deep Face Recognition:
        https://arxiv.org/abs/1801.07698v1

    Args:
        in_features: size of each input sample.
        out_features: size of each output sample.
        s: norm of input feature.
            Default: ``64.0``.
        m: margin.
            Default: ``0.5``.
        eps: operation accuracy.
            Default: ``1e-6``.

    Shape:
        - Input: :math:`(batch, H_{in})` where
          :math:`H_{in} = in\_features`.
        - Output: :math:`(batch, H_{out})` where
          :math:`H_{out} = out\_features`.

    Example:
        >>> layer = ArcFace(5, 10, s=1.31, m=0.5)
        >>> loss_fn = nn.CrossEntropyLoss()
        >>> embedding = torch.randn(3, 5, requires_grad=True)
        >>> target = torch.empty(3, dtype=torch.long).random_(10)
        >>> output = layer(embedding, target)
        >>> loss = loss_fn(output, target)
        >>> loss.backward()

    """

    def __init__(  # noqa: D107
        self,
        in_features: int,
        out_features: int,
        s: float = 64.0,
        m: float = 0.5,
        eps: float = 1e-6,
    ):
        super(ArcFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.threshold = math.pi - m
        self.eps = eps

        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def __repr__(self) -> str:
        """Object representation."""
        rep = (
            "ArcFace("
            f"in_features={self.in_features},"
            f"out_features={self.out_features},"
            f"s={self.s},"
            f"m={self.m},"
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

        if target is None:
            return cos_theta

        theta = torch.acos(torch.clamp(cos_theta, -1.0 + self.eps, 1.0 - self.eps))

        one_hot = torch.zeros_like(cos_theta)
        one_hot.scatter_(1, target.view(-1, 1).long(), 1)

        mask = torch.where(theta > self.threshold, torch.zeros_like(one_hot), one_hot)

        logits = torch.cos(torch.where(mask.bool(), theta + self.m, theta))
        logits *= self.s

        return logits


class SubCenterArcFace(nn.Module):
    """Implementation of
    `Sub-center ArcFace: Boosting Face Recognition
    by Large-scale Noisy Web Faces`_.

    .. _Sub-center ArcFace\: Boosting Face Recognition \
        by Large-scale Noisy Web Faces:
        https://ibug.doc.ic.ac.uk/media/uploads/documents/eccv_1445.pdf

    Args:
        in_features: size of each input sample.
        out_features: size of each output sample.
        s: norm of input feature,
            Default: ``64.0``.
        m: margin.
            Default: ``0.5``.
        k: number of possible class centroids.
            Default: ``3``.
        eps (float, optional): operation accuracy.
            Default: ``1e-6``.

    Shape:
        - Input: :math:`(batch, H_{in})` where
          :math:`H_{in} = in\_features`.
        - Output: :math:`(batch, H_{out})` where
          :math:`H_{out} = out\_features`.

    Example:
        >>> layer = SubCenterArcFace(5, 10, s=1.31, m=0.35, k=2)
        >>> loss_fn = nn.CrosEntropyLoss()
        >>> embedding = torch.randn(3, 5, requires_grad=True)
        >>> target = torch.empty(3, dtype=torch.long).random_(10)
        >>> output = layer(embedding, target)
        >>> loss = loss_fn(output, target)
        >>> loss.backward()

    """

    def __init__(  # noqa: D107
        self,
        in_features: int,
        out_features: int,
        s: float = 64.0,
        m: float = 0.5,
        k: int = 3,
        eps: float = 1e-6,
    ):
        super(SubCenterArcFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.s = s
        self.m = m
        self.k = k
        self.eps = eps

        self.weight = nn.Parameter(torch.FloatTensor(k, in_features, out_features))
        nn.init.xavier_uniform_(self.weight)

        self.threshold = math.pi - self.m

    def __repr__(self) -> str:
        """Object representation."""
        rep = (
            "SubCenterArcFace("
            f"in_features={self.in_features},"
            f"out_features={self.out_features},"
            f"s={self.s},"
            f"m={self.m},"
            f"k={self.k},"
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
            where ``C`` is a number of classes.
        """
        feats = F.normalize(input).unsqueeze(0).expand(self.k, *input.shape)  # k*b*f
        wght = F.normalize(self.weight, dim=1)  # k*f*c
        cos_theta = torch.bmm(feats, wght)  # k*b*f
        cos_theta = torch.max(cos_theta, dim=0)[0]  # b*f
        theta = torch.acos(torch.clamp(cos_theta, -1.0 + self.eps, 1.0 - self.eps))

        if target is None:
            return cos_theta

        one_hot = torch.zeros_like(cos_theta)
        one_hot.scatter_(1, target.view(-1, 1).long(), 1)

        selected = torch.where(theta > self.threshold, torch.zeros_like(one_hot), one_hot)

        logits = torch.cos(torch.where(selected.bool(), theta + self.m, theta))
        logits *= self.s

        return logits


__all__ = ["ArcFace", "SubCenterArcFace"]
