import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    def __init__(self, smooth=0, eps=1e-7):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.eps = eps

    def forward(self, output, target):
        return 1 - (2 * torch.sum(output * target) + self.smooth) / (
                torch.sum(output) + torch.sum(target) + self.smooth + self.eps)


class MixedDiceBCELoss(nn.Module):
    def __init__(self, dice_weight=0.2, dice_loss=None,
                        bce_weight=0.9, bce_loss=None,
                        smooth=0, dice_activation='sigmoid'):
        super(MixedDiceBCELoss, self).__init__()
        self.dice_weight = dice_weight
        self.dice_loss = dice_loss
        self.bce_weight = bce_weight
        self.bce_loss = bce_loss
        self.smooth = smooth
        self.dice_activation = dice_activation

    def forward(self, output, target):
        num_classes = output.size(1)
        target = target[:, :num_classes, :, :].long()
        if self.bce_loss is None:
            bce_loss = nn.BCEWithLogitsLoss()
        if self.dice_loss is None:
            dice_loss = multiclass_dice_loss
        return self.dice_weight * dice_loss(output, target, self.smooth, self.dice_activation) + self.bce_weight * bce_loss(output, target)


def mixed_dice_bce_loss(output, target, dice_weight=0.2, dice_loss=None,
                        bce_weight=0.9, bce_loss=None,
                        smooth=0, dice_activation='sigmoid'):

    num_classes = output.size(1)
    target = target[:, :num_classes, :, :].long()
    if bce_loss is None:
        bce_loss = nn.BCEWithLogitsLoss()
    if dice_loss is None:
        dice_loss = multiclass_dice_loss
    return dice_weight * dice_loss(output, target, smooth, dice_activation) + bce_weight * bce_loss(output, target)


def multiclass_dice_loss(output, target, smooth=0, activation='softmax'):
    """Calculate Dice Loss for multiple class output.
    Args:
        output (torch.Tensor): Model output of shape (N x C x H x W).
        target (torch.Tensor): Target of shape (N x H x W).
        smooth (float, optional): Smoothing factor. Defaults to 0.
        activation (string, optional): Name of the activation function, softmax or sigmoid. Defaults to 'softmax'.
    Returns:
        torch.Tensor: Loss value.
    """
    if activation == 'softmax':
        activation_nn = torch.nn.Softmax2d()
    elif activation == 'sigmoid':
        activation_nn = torch.nn.Sigmoid()
    else:
        raise NotImplementedError('only sigmoid and softmax are implemented')

    loss = 0
    dice = DiceLoss(smooth=smooth)
    output = activation_nn(output)
    num_classes = output.size(1)
    target.data = target.data.float()
    for class_nr in range(num_classes):
        loss += dice(output[:, class_nr, :, :], target[:, class_nr, :, :])
    return loss / num_classes


def where(cond, x_1, x_2):
    cond = cond.long()
    return (cond * x_1) + ((1 - cond) * x_2)