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
        self,
        in_features: int,
        out_features: int,
        s: float = 64.0,
        m: float = 0.35,
    ):
        super(CosFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m

        self.weight = nn.Parameter(
            torch.FloatTensor(out_features, in_features)
        )
        nn.init.xavier_uniform_(self.weight)

    def __repr__(self) -> str:
        """Object representation."""
        rep = "CosFace(in_features={},out_features={},s={},m={})".format(
            self.in_features, self.out_features, self.s, self.m
        )
        return rep

    def forward(
        self, input: torch.Tensor, target: torch.LongTensor
    ) -> torch.Tensor:
        """
        Args:
            input: input features,
                expected shapes ``BxF`` where ``B``
                is batch dimension and ``F`` is an
                input feature dimension.
            target: target classes,
                expected shapes ``B`` where
                ``B`` is batch dimension.

        Returns:
            tensor (logits) with shapes ``BxC``
            where ``C`` is a number of classes
            (out_features).
        """
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        phi = cosine - self.m

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, target.view(-1, 1).long(), 1)

        logits = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        logits *= self.s

        return logits
