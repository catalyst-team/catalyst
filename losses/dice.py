import torch
import torch.nn as nn
import torch.nn.functional as F


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
            raise NotImplementedError('only sigmoid and softmax are implemented')

    def forward(self, outputs, targets):
        outputs = self.activation_nn(outputs)
        return 1 - (2 * torch.sum(outputs * targets) + self.smooth) / (
                torch.sum(outputs) + torch.sum(targets) + self.smooth + self.eps)


class dice_bce_loss(nn.Module):
    def __init__(self, batch=True):
        super(dice_bce_loss, self).__init__()
        self.batch = batch
        self.bce_loss = nn.BCEWithLogitsLoss()

    def soft_dice_coeff(self, y_pred, y_true):
        y_pred = F.sigmoid(y_pred)
        smooth = 0.0  # may change
        if self.batch:
            i = torch.sum(y_true)
            j = torch.sum(y_pred)
            intersection = torch.sum(y_true * y_pred)
        else:
            i = y_true.sum(1).sum(1).sum(1)
            j = y_pred.sum(1).sum(1).sum(1)
            intersection = (y_true * y_pred).sum(1).sum(1).sum(1)
        score = (2. * intersection + smooth) / (i + j + smooth)
        # score = (intersection + smooth) / (i + j - intersection + smooth)#iou
        return score.mean()

    def soft_dice_loss(self, y_true, y_pred):
        loss = 1 - self.soft_dice_coeff(y_true, y_pred)
        return loss

    def __call__(self, y_pred, y_true):
        a = self.bce_loss(y_pred, y_true)
        b = self.soft_dice_loss(y_pred, y_true)
        return a + b