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
