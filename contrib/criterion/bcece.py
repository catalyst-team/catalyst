import torch
import torch.nn as nn


class BCESoftmaxLoss(nn.Module):
    def __init__(
        self,
        bce_indices,
        ce_indices,
        bce2ce_alpha=0.5,
        reduction="elementwise_mean"
    ):
        super().__init__()
        self.reduction = reduction
        self.bce = nn.BCEWithLogitsLoss()
        self.ce = nn.CrossEntropyLoss()
        self.bce_indices = bce_indices
        self.ce_indices = ce_indices
        self.alpha = bce2ce_alpha

    def forward(self, y_pred, y_true):
        bce_loss = self.bce(
            y_pred[:, self.bce_indices, ...], y_true[:, self.bce_indices, ...]
        )
        y_true_ce = torch.argmax(
            y_true[:, self.ce_indices, ...], dim=1, keepdim=False
        ).long()
        ce_loss = self.ce(y_pred[:, self.ce_indices, ...], y_true_ce)
        loss = self.alpha * bce_loss + (1.0 - self.alpha) * ce_loss

        if self.reduction == "elementwise_mean":
            loss = torch.mean(loss)
        elif self.reduction == "sum":
            loss = torch.sum(loss)

        return loss
