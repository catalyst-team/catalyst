from functools import partial

import torch.nn as nn
from catalyst.dl.utils import criterion


class DiceLoss(nn.Module):
    def __init__(
        self,
        eps: float = 1e-7,
        threshold: float = None,
        activation: str = "Sigmoid"
    ):
        super().__init__()

        self.loss_fn = partial(
            criterion.dice,
            eps=eps,
            threshold=threshold,
            activation=activation)

    def forward(self, logits, targets):
        dice = self.loss_fn(logits, targets)
        return 1 - dice


class BCEDiceLoss(nn.Module):
    def __init__(
        self,
        eps: float = 1e-7,
        threshold: float = None,
        activation: str = "Sigmoid"
    ):
        super().__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss(
            eps=eps,
            threshold=threshold,
            activation=activation
        )

    def forward(self, outputs, targets):
        dice = self.dice_loss(outputs, targets)
        bce = self.bce_loss(outputs, targets)
        loss = dice + bce
        return loss
