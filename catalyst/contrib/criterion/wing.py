from functools import partial

import torch.nn as nn
from catalyst.dl import losses


class WingLoss(nn.Module):
    def __init__(
        self,
        width: int = 5,
        curvature: float = 0.5,
        reduction: str = "mean"
    ):
        super().__init__()
        self.loss_fn = partial(
            losses.wing_loss,
            width=width,
            curvature=curvature,
            reduction=reduction)

    def forward(self, outputs, targets):
        loss = self.loss_fn(outputs, targets)
        return loss
