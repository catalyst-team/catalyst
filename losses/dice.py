import torch
import torch.nn as nn


class BCEDiceLoss(nn.Module):
    def __init__(self, batch=True, activation='sigmoid'):
        super(BCEDiceLoss, self).__init__()
        self.batch = batch
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss(activation)

    def forward(self, outputs, targets):
        a = self.bce_loss(outputs, targets)
        b = self.dice_loss(outputs, targets)
        return a + b


class DiceLoss(nn.Module):
    def __init__(self, smooth=0, eps=1e-7, activation='sigmoid'):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.eps = eps
        if activation == 'softmax':
            self.activation_nn = torch.nn.Softmax2d()
        elif activation == 'sigmoid':
            self.activation_nn = torch.nn.Sigmoid()
        else:
            raise NotImplementedError(
                'only sigmoid and softmax are implemented'
            )

    def forward(self, outputs, targets):
        outputs = self.activation_nn(outputs)
        return 1 - (2 * torch.sum(outputs * targets) + self.smooth) / \
            (torch.sum(outputs) + torch.sum(targets) + self.smooth + self.eps)
