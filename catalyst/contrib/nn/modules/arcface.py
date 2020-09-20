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
    ):
        """
        Args:
            in_features (int): size of each input sample.
            out_features (int): size of each output sample.
            s (float, optional): norm of input feature,
                Default: ``64.0``.
            m (float, optional): margin.
                Default: ``0.5``.
        """
        super(ArcFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

        self.weight = nn.Parameter(
            torch.FloatTensor(out_features, in_features)
        )
        nn.init.xavier_uniform_(self.weight)

    def __repr__(self) -> str:
        return "ArcFace(in_features={},out_features={},s={},m={})".format(
            self.in_features, self.out_features, self.s, self.m
        )

    def forward(
        self, input: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            input (torch.Tensor): input features,
                expected shapes BxF.
            target (torch.Tensor): target classes,
                expected shapes B.

        Returns:
            torch.Tensor with loss value.
        """
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m

        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros(cosine.size()).to(input.device)
        one_hot.scatter_(1, target.view(-1, 1).long(), 1)
        logits = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        logits *= self.s

        return logits
