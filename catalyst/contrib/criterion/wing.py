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
        self.metric_fn = losses.wing_loss

        self.width = width
        self.curvature = curvature
        self.reduction = reduction

    def forward(self, outputs, targets):
        wing = self.metric_fn(
            outputs,
            targets,
            width=self.width,
            curvature=self.curvature,
            reduction=self.reduction,
        )
        return wing
