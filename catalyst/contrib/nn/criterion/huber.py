# flake8: noqa
# @TODO: code formatting issue for 20.07 release
import torch
from torch import nn


class HuberLoss(nn.Module):
    """@TODO: Docs. Contribution is welcome."""

    def __init__(self, clip_delta=1.0, reduction="mean"):
        """@TODO: Docs. Contribution is welcome."""
        super().__init__()
        self.clip_delta = clip_delta
        self.reduction = reduction or "none"

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, weights=None) -> torch.Tensor:
        """@TODO: Docs. Contribution is welcome."""
        td_error = y_true - y_pred
        td_error_abs = torch.abs(td_error)
        quadratic_part = torch.clamp(td_error_abs, max=self.clip_delta)
        linear_part = td_error_abs - quadratic_part
        loss = 0.5 * quadratic_part ** 2 + self.clip_delta * linear_part

        if weights is not None:
            loss = torch.mean(loss * weights, dim=1)
        else:
            loss = torch.mean(loss, dim=1)

        if self.reduction == "mean":
            loss = torch.mean(loss)
        elif self.reduction == "sum":
            loss = torch.sum(loss)

        return loss


__all__ = ["HuberLoss"]
