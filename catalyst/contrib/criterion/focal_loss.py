from .functional import sigmoid_focal_loss
from torch.nn.modules.loss import _Loss


class FocalLossBinary(_Loss):
    def __init__(self, alpha=0.5, gamma=2, ignore=None):
        """
        Compute focal loss for binary classification problem.
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore = ignore

    def forward(self, logits, targets):
        """

        :param logits: [bs]
        :param targets: [bs]
        :return:
        """
        targets = targets.view(-1)
        logits = logits.view(-1)

        if self.ignore is not None:
            # Filter predictions with ignore label from loss computation
            not_ignored = targets != self.ignore
            logits = logits[not_ignored]
            targets = targets[not_ignored]

        loss = sigmoid_focal_loss(
            logits,
            targets,
            gamma=self.gamma,
            alpha=self.alpha
        )

        return loss


class FocalLossMultiClass(_Loss):
    def __init__(self, alpha=0.5, gamma=2, ignore=None):
        """
        Compute focal loss for multi-class problem.
        Ignores targets having -1 label
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore = ignore

    def forward(self, logits, targets):
        """

        :param logits: [bs; num_classes]
        :param targets: [bs]
        :return:
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

            loss += sigmoid_focal_loss(
                cls_label_input,
                cls_label_target,
                gamma=self.gamma,
                alpha=self.alpha
            )

        return loss


class FocalLossMultiLabel(_Loss):
    def __init__(self, alpha=0.5, gamma=2, ignore=None):
        """
        Compute focal loss for multi-label problem.
        Ignores targets having -1 label
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore = ignore

    def forward(self, logits, targets):
        """

        :param logits: [bs; num_classes]
        :param targets: [bs; num_classes]
        :return:
        """
        num_classes = logits.size(1)
        loss = 0

        for cls in range(num_classes):
            # Filter anchors with -1 label from loss computation
            if cls == self.ignore:
                continue

            cls_label_target = targets[..., cls].long()
            cls_label_input = logits[..., cls]

            loss += sigmoid_focal_loss(
                cls_label_input,
                cls_label_target,
                gamma=self.gamma,
                alpha=self.alpha
            )

        return loss


__all__ = ["FocalLossBinary", "FocalLossMultiClass", "FocalLossMultiLabel"]
