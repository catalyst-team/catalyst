import torch
from torch import nn
from torch.nn import functional as F


class NaiveCrossEntropyLoss(nn.Module):
    """@TODO: Docs. Contribution is welcome."""

    def __init__(self, size_average=True):
        """@TODO: Docs. Contribution is welcome."""
        super().__init__()
        self.size_average = size_average

    def forward(
        self, input: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Calculates loss between ``input`` and ``target`` tensors.

        Args:
            input (torch.Tensor): input tensor of shape ...
            target (torch.Tensor): target tensor of shape ...

        @TODO: Docs (add shapes). Contribution is welcome.
        """
        assert input.size() == target.size()
        input = F.log_softmax(input)
        loss = -torch.sum(input * target)
        loss = loss / input.size()[0] if self.size_average else loss
        return loss


class SymmetricCrossEntropyLoss(nn.Module):
    """The Symmetric Cross Entropy loss.

    It has been proposed in `Symmetric Cross Entropy for Robust Learning
    with Noisy Labels`_.

    .. _Symmetric Cross Entropy for Robust Learning with Noisy Labels:
        https://arxiv.org/abs/1908.06112
    """

    def __init__(self, alpha: float = 1.0, beta: float = 1.0):
        """
        Args:
            alpha(float):
                corresponds to overfitting issue of CE
            beta(float):
                corresponds to flexible exploration on the robustness of RCE
        """
        super(SymmetricCrossEntropyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(
        self, input: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Calculates loss between ``input`` and ``target`` tensors.

        Args:
            input (torch.Tensor): input tensor of size
                (batch_size, num_classes)
            target (torch.Tensor): target tensor of size (batch_size), where
                values of a vector correspond to class index
        """
        num_classes = input.shape[1]
        target_one_hot = F.one_hot(target, num_classes).float()
        assert target_one_hot.shape == input.shape

        input = torch.clamp(input, min=1e-7, max=1.0)
        target_one_hot = torch.clamp(target_one_hot, min=1e-4, max=1.0)

        cross_entropy = (
            -torch.sum(target_one_hot * torch.log(input), dim=1)
        ).mean()
        reverse_cross_entropy = (
            -torch.sum(input * torch.log(target_one_hot), dim=1)
        ).mean()
        loss = self.alpha * cross_entropy + self.beta * reverse_cross_entropy
        return loss


class MaskCrossEntropyLoss(torch.nn.CrossEntropyLoss):
    """@TODO: Docs. Contribution is welcome."""

    def __init__(
        self,
        *args,
        target_name: str = "targets",
        mask_name: str = "mask",
        **kwargs
    ):
        """@TODO: Docs. Contribution is welcome."""
        super().__init__(*args, **kwargs)
        self.target_name = target_name
        self.mask_name = mask_name
        self.reduction = "none"

    def forward(
        self, input: torch.Tensor, target_mask: torch.Tensor
    ) -> torch.Tensor:
        """Calculates loss between ``input`` and ``target_mask`` tensors.

        @TODO: Docs. Contribution is welcome.
        """
        target = target_mask[self.target_name]
        mask = target_mask[self.mask_name]

        loss = super().forward(input, target)
        loss = torch.mean(loss[mask == 1])
        return loss


__all__ = [
    "MaskCrossEntropyLoss",
    "SymmetricCrossEntropyLoss",
    "NaiveCrossEntropyLoss",
]
