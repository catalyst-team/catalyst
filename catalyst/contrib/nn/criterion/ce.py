# flake8: noqa
# TODO: update docs and shapes
import torch
from torch import nn
from torch.nn import functional as F


class NaiveCrossEntropyLoss(nn.Module):
    """@TODO: Docs. Contribution is welcome."""

    def __init__(self, size_average=True):
        """@TODO: Docs. Contribution is welcome."""
        super().__init__()
        self.size_average = size_average

    def forward(self, input_: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculates loss between ``input_`` and ``target`` tensors.

        Args:
            input_: input tensor of shape ...
            target: target tensor of shape ...

        @TODO: Docs (add shapes). Contribution is welcome.
        """
        assert input_.size() == target.size()
        input_ = F.log_softmax(input_)
        loss = -torch.sum(input_ * target)
        loss = loss / input_.size()[0] if self.size_average else loss
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

    def forward(self, input_: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculates loss between ``input_`` and ``target`` tensors.

        Args:
            input_: input tensor of size
                (batch_size, num_classes)
            target: target tensor of size (batch_size), where
                values of a vector correspond to class index

        Returns:
            torch.Tensor: computed loss
        """
        num_classes = input_.shape[1]
        target_one_hot = F.one_hot(target, num_classes).float()
        assert target_one_hot.shape == input_.shape

        input_ = torch.clamp(input_, min=1e-7, max=1.0)
        target_one_hot = torch.clamp(target_one_hot, min=1e-4, max=1.0)

        cross_entropy = (-torch.sum(target_one_hot * torch.log(input_), dim=1)).mean()
        reverse_cross_entropy = (-torch.sum(input_ * torch.log(target_one_hot), dim=1)).mean()
        loss = self.alpha * cross_entropy + self.beta * reverse_cross_entropy
        return loss


class MaskCrossEntropyLoss(nn.Module):
    """@TODO: Docs. Contribution is welcome."""

    def __init__(self, *args, **kwargs):
        """@TODO: Docs. Contribution is welcome."""
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss(*args, **kwargs, reduction="none")

    def forward(
        self, logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculates loss between ``logits`` and ``target`` tensors.

        Args:
            logits: model logits
            target: true targets
            mask: targets mask

        Returns:
            torch.Tensor: computed loss
        """
        loss = self.ce_loss.forward(logits, target)
        loss = torch.mean(loss[mask == 1])
        return loss


__all__ = [
    "MaskCrossEntropyLoss",
    "SymmetricCrossEntropyLoss",
    "NaiveCrossEntropyLoss",
]
