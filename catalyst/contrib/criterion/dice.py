import torch.nn as nn
from catalyst.dl import metrics


class DiceLoss(nn.Module):
    def __init__(
        self,
        eps: float = 1e-7,
        threshold: float = None,
        activation: str = "sigmoid"
    ):
        super(DiceLoss, self).__init__()
        self.eps = eps
        self.threshold = threshold
        self.activation = activation

    def forward(self, outputs, targets):
        dice = metrics.dice(
            outputs, targets,
            eps=self.eps,
            threshold=self.threshold,
            activation=self.activation
        )

        return 1 - dice


class BCEDiceLoss(nn.Module):
    def __init__(
        self,
        eps: float = 1e-7,
        threshold: float = None,
        activation: str = "sigmoid"
    ):
        super().__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss(
            eps=eps,
            threshold=threshold,
            activation=activation
        )

    def forward(self, outputs, targets):
        dice = self.dice_loss.forward(outputs, targets)
        bce = self.bce_loss(outputs, targets)
        loss = dice + bce
        return loss
