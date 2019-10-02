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
            activation=activation
        )

    def forward(self, logits, targets):
        dice = self.loss_fn(logits, targets)
        return 1 - dice


class BCEDiceLoss(nn.Module):
    def __init__(
            self,
            eps: float = 1e-7,
            threshold: float = None,
            activation: str = "Sigmoid",
            bce_weight: float = 0.5,
            dice_weight: float = 0.5,
    ):
        super().__init__()

        if bce_weight == 0 and dice_weight == 0:
            raise ValueError(
                "Both bce_wight and dice_weight cannot be "
                "equal to 0 at the same time."
            )

        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

        if self.bce_weight != 0:
            self.bce_loss = nn.BCEWithLogitsLoss()

        if self.dice_weight != 0:
            self.dice_loss = DiceLoss(
                eps=eps, threshold=threshold, activation=activation
            )

    def forward(self, outputs, targets):
        if self.bce_weight == 0:
            return self.dice_weight * self.dice_loss(outputs, targets)
        if self.dice_weight == 0:
            return self.bce_weight * self.bce_loss(outputs, targets)

        dice = self.dice_weight * self.dice_loss(outputs, targets)
        bce = self.bce_weight * self.bce_loss(outputs, targets)
        return dice + bce
