import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftMax(nn.Module):
    """Implementation of SoftMax head for metric learning.

    Example:
        >>> layer = SoftMax()
        >>> loss_fn = nn.CrosEntropyLoss()
        >>> embedding = torch.randn(3, 5, requires_grad=True)
        >>> target = torch.empty(3, dtype=torch.long).random_(5)
        >>> output = layer(embedding, target)
        >>> loss = loss_fn(output, target)
        >>> loss.backward()

    """

    def __init__(self, in_features: int, num_classes: int):
        """
        Args:
            in_features (int): size of each input sample.
            num_classes (int): size of each output sample.
        """
        super(SoftMax, self).__init__()
        self.in_features = in_features
        self.out_features = num_classes
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, in_features))
        self.bias = nn.Parameter(torch.FloatTensor(num_classes))

        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def __repr__(self) -> str:
        """Object representation."""
        return "SoftMax(in_features={},out_features={})".format(
            self.in_features, self.out_features
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
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
        return F.linear(input, self.weight, self.bias)
