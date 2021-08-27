import torch
from torch import nn


class NTXentLoss(nn.Module):
    """A Contrastive embedding loss.

    It has been proposed in `A Simple Framework
    for Contrastive Learning of Visual Representations`_.

    Example:

    .. code-block:: python

        import torch
        from torch.nn import functional as F
        from catalyst.contrib.nn import NTXentLoss

        embeddings_left = F.normalize(torch.rand(256, 64, requires_grad=True))
        embeddings_right = F.normalize(torch.rand(256, 64, requires_grad=True))
        criterion = NTXentLoss(tau = 0.1)
        criterion(embeddings_left, embeddings_right)

    .. _`A Simple Framework for Contrastive Learning of Visual Representations`:
        https://arxiv.org/abs/2103.03230
    """

    def __init__(self, tau: float) -> None:
        """

        Args:
            tau: tau to use
        """
        super().__init__()
        self.tau = tau
        self.cosineSim = nn.CosineSimilarity()

    def forward(self, features1: torch.Tensor, features2: torch.Tensor) -> torch.Tensor:
        """

        Args:
            features1: batch with samples features of shape
                [bs; feature_len]
            features2: batch with samples features of shape
                [bs; feature_len]

        Returns:
            torch.Tensor: NTXent loss
        """
        assert (
            features1.shape == features2.shape
        ), f"Invalid shape of input features: {features1.shape} and {features2.shape}"
        bs = features1.shape[0]

        pos_loss = self.cosineSim(features1, features2).sum(dim=0) / self.tau
        list_neg_loss = [
            torch.exp(self.cosineSim(features1, torch.roll(features2, i, 1)) / self.tau)
            for i in range(0, bs)
        ]
        # todo try different places for temparature
        neg_loss = torch.log(torch.stack(list_neg_loss, dim=0).sum(dim=0)).sum(dim=0)

        loss = -pos_loss + neg_loss
        return loss
