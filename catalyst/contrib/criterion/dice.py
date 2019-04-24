import torch.nn as nn
from catalyst.dl import metrics


class DiceLoss(nn.Module):
    def __init__(self, eps=1e-7, activation="sigmoid"):
        super(DiceLoss, self).__init__()
        self.activation = activation
        self.eps = eps

    def forward(self, outputs, targets):
        dice = metrics.dice(
            outputs, targets, eps=self.eps, activation=self.activation
        )

        return 1 - dice


class BCEDiceLoss(nn.Module):
    def __init__(self, eps=1e-7, activation="sigmoid"):
        super(BCEDiceLoss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss(eps=eps, activation=activation)

    def forward(self, outputs, targets):
        return self.bce_loss(outputs,
                             targets) + self.dice_loss(outputs, targets)
