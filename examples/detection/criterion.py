import torch
import torch.nn as nn
import torch.nn.functional as F


class SSDCriterion(nn.Module):
    def __init__(self, num_classes, ignore_class=0):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_class = ignore_class

    def _hard_negative_mining(self, cls_loss, pos):
        """Return negative indices that is 3x the number as positive indices.

        Args:
            cls_loss: (torch.Tensor) cross entropy loss between `cls_preds` and `cls_targets`.
                Expected shape [B, M] where B - batch, M - anchors.
            pos: (torch.Tensor) positive class mask.
                Expected shape [B, M] where B - batch, M - anchors.

        Return:
            (torch.Tensor) negative indices, sized [N,#anchors].
        """
        cls_loss = cls_loss * (pos.float() - 1)

        _, idx = cls_loss.sort(1)  # sort by negative losses
        _, rank = idx.sort(1)  # [B, M]

        num_neg = 3 * pos.sum(1)  # [B,]
        neg = rank < num_neg[:, None]  # [B, M]
        return neg

    def forward(self, loc_preds, loc_targets, cls_preds, cls_targets):
        """Compute loss between (loc_preds, loc_targets) and (cls_preds, cls_targets).
        Loss:
            loss = SmoothL1Loss(loc_preds, loc_targets) + CrossEntropyLoss(cls_preds, cls_targets).

        Args:
            loc_preds: (torch.Tensor) predicted locations.
                Expected shapes - [B, M, 4] where B - batch, M - anchors.
            loc_targets: (torch.Tensor) encoded target locations.
                Expected shapes - [B, M, 4] where B - batch, M - anchors.
            cls_preds: (torch.Tensor) predicted class confidences.
                Expected shapes - [B, M, CLASS] where B - batch, M - anchors, CLASS - number of classes.
            cls_targets: (torch.LongTensor) encoded target labels.
                Expected shapes - [B, M] where B - batch, M - anchors.

        Returns:
            regression loss and classification loss
        """
        pos = cls_targets != self.ignore_class  # not background
        batch_size = pos.size(0)
        num_pos = pos.sum().item()

        mask = pos.unsqueeze(2).expand_as(loc_preds)  # [B, M, 4]
        loc_loss = F.smooth_l1_loss(loc_preds[mask], loc_targets[mask], reduction="sum")

        cls_loss = F.cross_entropy(
            cls_preds.view(-1, self.num_classes), cls_targets.view(-1), reduction="none"
        )  # [B * M,]
        cls_loss = cls_loss.view(batch_size, -1)
        cls_loss[cls_targets == self.ignore_class] = 0  # set ignored loss to 0
        neg = self._hard_negative_mining(cls_loss, pos)  # [B, M]
        cls_loss = cls_loss[pos | neg].sum()

        loc_loss = loc_loss / num_pos
        cls_loss = cls_loss / num_pos

        return loc_loss, cls_loss


def reg_loss(regr, gt_regr, mask):
    """L1 regression loss

    Arguments:
        regr (batch x max_objects x dim)
        gt_regr (batch x max_objects x dim)
        mask (batch x max_objects)
    """
    num = mask.float().sum()
    # print(gt_regr.size())
    mask = mask.sum(1).unsqueeze(1).expand_as(gt_regr)
    # print(mask.size())

    regr = regr * mask
    gt_regr = gt_regr * mask

    regr_loss = F.smooth_l1_loss(regr, gt_regr, size_average=False)
    regr_loss = regr_loss / (num + 1e-4)
    return regr_loss


def neg_loss(pred, gt):
    """Modified focal loss. Exactly the same as CornerNet.
    Runs faster and costs a little bit more memory

    Arguments:
        pred (batch x c x h x w)
        gt_regr (batch x c x h x w)
    """
    pred = pred.unsqueeze(1).float()
    gt = gt.unsqueeze(1).float()

    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()
    neg_weights = torch.pow(1 - gt, 4)

    loss = 0

    pos_loss = torch.log(pred + 1e-12) * torch.pow(1 - pred, 3) * pos_inds
    neg_loss = torch.log(1 - pred + 1e-12) * torch.pow(pred, 3) * neg_weights * neg_inds

    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos

    return loss


class CenterNetCriterion(nn.Module):
    def __init__(
        self, num_classes=1, mask_loss_weight=1.0, regr_loss_weight=1.0, size_average=True
    ):
        super().__init__()
        self.num_classes = num_classes
        self.mask_loss_weight = mask_loss_weight
        self.regr_loss_weight = regr_loss_weight
        self.size_average = size_average

    def forward(self, predicted_heatmap, predicted_regr, target_heatmap, target_regr):
        pred_mask = torch.sigmoid(predicted_heatmap)
        mask_loss = neg_loss(pred_mask, target_heatmap)
        mask_loss *= self.mask_loss_weight

        regr_loss = (
            torch.abs(predicted_regr - target_regr).sum(1)[:, None, :, :] * target_heatmap
        ).sum()  # .sum(1).sum(1).sum(1)
        regr_loss = regr_loss / target_heatmap.sum()  # .sum(1).sum(1).sum(1)
        regr_loss *= self.regr_loss_weight

        loss = mask_loss + regr_loss
        if not self.size_average:
            loss *= predicted_heatmap.shape[0]

        return loss, mask_loss, regr_loss
