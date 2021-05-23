from functools import partial

from torch.nn.modules.loss import _Loss  # noqa: WPS450

from catalyst import metrics


class FocalLossBinary(_Loss):
    """Compute focal loss for binary classification problem.

    It has been proposed in `Focal Loss for Dense Object Detection`_ paper.

    .. _Focal Loss for Dense Object Detection: https://arxiv.org/abs/1708.02002
    """

    def __init__(
        self,
        ignore: int = None,
        reduced: bool = False,
        gamma: float = 2.0,
        alpha: float = 0.25,
        threshold: float = 0.5,
        reduction: str = "mean",
    ):
        """@TODO: Docs. Contribution is welcome."""
        super().__init__()
        self.ignore = ignore

        if reduced:
            self.loss_fn = partial(
                metrics.reduced_focal_loss, gamma=gamma, threshold=threshold, reduction=reduction,
            )
        else:
            self.loss_fn = partial(
                metrics.sigmoid_focal_loss, gamma=gamma, alpha=alpha, reduction=reduction,
            )

    def forward(self, logits, targets):
        """
        Args:
            logits: [bs; ...]
            targets: [bs; ...]

        Returns:
            computed loss
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
    """Compute focal loss for multiclass problem. Ignores targets having -1 label.

    It has been proposed in `Focal Loss for Dense Object Detection`_ paper.

    .. _Focal Loss for Dense Object Detection: https://arxiv.org/abs/1708.02002
    """

    def forward(self, logits, targets):
        """
        Args:
            logits: [bs; num_classes; ...]
            targets: [bs; ...]

        Returns:
            computed loss
        """
        num_classes = logits.size(1)
        loss = 0
        targets = targets.view(-1)
        logits = logits.view(-1, num_classes)

        # Filter anchors with -1 label from loss computation
        if self.ignore is not None:
            not_ignored = targets != self.ignore

        for class_id in range(num_classes):
            cls_label_target = (targets == (class_id + 0)).long()  # noqa: WPS345
            cls_label_input = logits[..., class_id]

            if self.ignore is not None:
                cls_label_target = cls_label_target[not_ignored]
                cls_label_input = cls_label_input[not_ignored]

            loss += self.loss_fn(cls_label_input, cls_label_target)

        return loss


# @TODO: check
# class FocalLossMultiLabel(_Loss):
#     """Compute focal loss for multilabel problem.
#     Ignores targets having -1 label.
#
#     It has been proposed in `Focal Loss for Dense Object Detection`_ paper.
#
#     @TODO: Docs (add `Example`). Contribution is welcome.
#
#     .. _Focal Loss for Dense Object Detection:
#         https://arxiv.org/abs/1708.02002
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
