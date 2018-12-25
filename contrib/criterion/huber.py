import torch
import torch.nn as nn


class HuberLoss(nn.Module):
    def __init__(self, clip_delta=1.0, reduction="elementwise_mean"):
        super(HuberLoss, self).__init__()
        self.clip_delta = clip_delta
        self.reduction = reduction or "none"

    def forward(self, y_pred, y_true, weights=None):
        td_error = y_true - y_pred
        td_error_abs = torch.abs(td_error)
        quadratic_part = torch.clamp(td_error_abs, max=self.clip_delta)
        linear_part = td_error_abs - quadratic_part
        loss = 0.5 * quadratic_part**2 + self.clip_delta * linear_part

        if weights is not None:
            loss = torch.mean(loss * weights, dim=1)
        else:
            loss = torch.mean(loss, dim=1)

        if self.reduction == "elementwise_mean":
            loss = torch.mean(loss)
        elif self.reduction == "sum":
            loss = torch.sum(loss)

        return loss
