import torch
import torch.nn as nn
import torch.nn.functional as F


class CosFace(nn.Module):
    """Implementation of CosFace loss for metric learning.

    .. _CosFace: Large Margin Cosine Loss for Deep Face Recognition:
        https://arxiv.org/abs/1801.09414

    Example:
        >>> layer = CosFaceLoss(5, 10, s=1.31, m=0.1)
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
        m: float = 0.35,
    ):
        """
        Args:
            in_features (int): size of each input sample.
            out_features (int): size of each output sample.
            s (float, optional): norm of input feature,
                Default: ``64.0``.
            m (float, optional): margin.
                Default: ``0.35``.
        """
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
        return "CosFace(in_features={},out_features={},s={},m={})".format(
            self.in_features, self.out_features, self.s, self.m
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
            logits tensor with shapes ``BxC``
            where C is a number of classes.
        """
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        phi = cosine - self.m
        one_hot = torch.zeros(cosine.size()).to(input.device)
        one_hot.scatter_(1, target.view(-1, 1).long(), 1)
        logits = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        logits *= self.s

        return logits
