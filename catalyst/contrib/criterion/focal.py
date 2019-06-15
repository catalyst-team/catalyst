from functools import partial

from torch.nn.modules.loss import _Loss
from catalyst.dl.utils import criterion


class FocalLossBinary(_Loss):
    def __init__(
        self,
        ignore: int = None,
        reduced: bool = False,
        gamma: float = 2.0,
        alpha: float = 0.25,
        threshold: float = 0.5,
        reduction: str = "mean",
    ):
        """
        Compute focal loss for binary classification problem.
        """
        super().__init__()
        self.ignore = ignore

        if reduced:
            self.loss_fn = partial(
                criterion.reduced_focal_loss,
                gamma=gamma,
                threshold=threshold,
                reduction=reduction)
        else:
            self.loss_fn = partial(
                criterion.sigmoid_focal_loss,
                gamma=gamma,
                alpha=alpha,
                reduction=reduction)

    def forward(self, logits, targets):
        """
        Args:
            logits: [bs; ...]
            targets: [bs; ...]
        """
        targets = targets.view(-1)
        logits = logits.view(-1)

        if self.ignore is not None:
            # Filter predictions with ignore label from loss computation
            not_ignored = targets != self.ignore
            logits = logits[not_ignored]
            targets = targets[not_ignored]

        loss = self.loss_fn(logits, targets)

        return loss


class FocalLossMultiClass(FocalLossBinary):
    """
    Compute focal loss for multi-class problem.
    Ignores targets having -1 label
    """

    def forward(self, logits, targets):
        """
        Args:
            logits: [bs; num_classes; ...]
            targets: [bs; ...]
        """
        num_classes = logits.size(1)
        loss = 0
        targets = targets.view(-1)
        logits = logits.view(-1, num_classes)

        # Filter anchors with -1 label from loss computation
        if self.ignore is not None:
            not_ignored = targets != self.ignore

        for cls in range(num_classes):
            cls_label_target = (targets == (cls + 0)).long()
            cls_label_input = logits[..., cls]

            if self.ignore is not None:
                cls_label_target = cls_label_target[not_ignored]
                cls_label_input = cls_label_input[not_ignored]

            loss += self.loss_fn(cls_label_input, cls_label_target)

        return loss


# @TODO: check
# class FocalLossMultiLabel(_Loss):
#     """
#     Compute focal loss for multi-label problem.
#     Ignores targets having -1 label
#     """
#
#     def forward(self, logits, targets):
#         """
#         Args:
#             logits: [bs; num_classes]
#             targets: [bs; num_classes]
#         """
#         num_classes = logits.size(1)
#         loss = 0
#
#         for cls in range(num_classes):
#             # Filter anchors with -1 label from loss computation
#             if cls == self.ignore:
#                 continue
#
#             cls_label_target = targets[..., cls].long()
#             cls_label_input = logits[..., cls]
#
#             loss += self.loss_fn(cls_label_input, cls_label_target)
#
#         return loss


__all__ = ["FocalLossBinary", "FocalLossMultiClass"]
