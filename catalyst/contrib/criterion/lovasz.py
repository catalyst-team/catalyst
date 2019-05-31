"""
https://arxiv.org/abs/1705.08790
Lovasz-Softmax and Jaccard hinge loss in PyTorch
Maxim Berman 2018 ESAT-PSI KU Leuven (MIT License)
"""

from itertools import filterfalse as ifilterfalse

import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss


# --------------------------- HELPER FUNCTIONS ---------------------------


def isnan(x):
    return x != x


def mean(values, ignore_nan=False, empty=0):
    """
    Nanmean compatible with generators.
    """
    values = iter(values)
    if ignore_nan:
        values = ifilterfalse(isnan, values)
    try:
        n = 1
        acc = next(values)
    except StopIteration:
        if empty == "raise":
            raise ValueError("Empty mean")
        return empty
    for n, v in enumerate(values, 2):
        acc += v
    if n == 1:
        return acc
    return acc / n


def _lovasz_grad(gt_sorted):
    """
    Compute gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1:  # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


# ---------------------------- BINARY LOSSES -----------------------------


def _flatten_binary_scores(logits, targets, ignore=None):
    """
    Flattens predictions in the batch (binary case)
    Remove targets equal to "ignore"
    """
    logits = logits.view(-1)
    targets = targets.view(-1)
    if ignore is None:
        return logits, targets
    valid = (targets != ignore)
    logits_ = logits[valid]
    targets_ = targets[valid]
    return logits_, targets_


def _lovasz_hinge_flat(logits, targets):
    """
    Binary Lovasz hinge loss

    Args:
        logits: [P] Variable, logits at each prediction
            (between -iinfinity and +iinfinity)
        targets: [P] Tensor, binary ground truth targets (0 or 1)
        ignore: label to ignore
    """
    if len(targets) == 0:
        # only void pixels, the gradients should be 0
        return logits.sum() * 0.
    signs = 2. * targets.float() - 1.
    errors = (1. - logits * signs)
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    perm = perm.data
    gt_sorted = targets[perm]
    grad = _lovasz_grad(gt_sorted)
    loss = torch.dot(F.relu(errors_sorted), grad)
    return loss


def _lovasz_hinge(logits, targets, per_image=True, ignore=None):
    """
    Binary Lovasz hinge loss

    Args:
        logits: [B, H, W] Variable, logits at each pixel
            (between -infinity and +infinity)
        targets: [B, H, W] Tensor, binary ground truth masks (0 or 1)
        per_image: compute the loss per image instead of per batch
        ignore: void class id
    """
    if per_image:
        loss = mean(
            _lovasz_hinge_flat(
                *_flatten_binary_scores(
                    logit.unsqueeze(0),
                    target.unsqueeze(0),
                    ignore))
            for logit, target in zip(logits, targets))
    else:
        loss = _lovasz_hinge_flat(
            *_flatten_binary_scores(logits, targets, ignore))
    return loss


# --------------------------- MULTICLASS LOSSES ---------------------------


def _flatten_probabilities(probabilities, targets, ignore=None):
    """
    Flattens predictions in the batch
    """
    if probabilities.dim() == 3:
        # assumes output of a sigmoid layer
        B, H, W = probabilities.size()
        probabilities = probabilities.view(B, 1, H, W)
    B, C, H, W = probabilities.size()
    # B * H * W, C = P, C
    probabilities = probabilities.permute(0, 2, 3, 1).contiguous().view(-1, C)
    targets = targets.view(-1)
    if ignore is None:
        return probabilities, targets
    valid = (targets != ignore)
    probabilities_ = probabilities[valid.nonzero().squeeze()]
    targets_ = targets[valid]
    return probabilities_, targets_


def _lovasz_softmax_flat(probabilities, targets, classes="present"):
    """
    Multi-class Lovasz-Softmax loss

    Args:
        probabilities: [P, C]
            class probabilities at each prediction (between 0 and 1)
        targets: [P] ground truth targets (between 0 and C - 1)
        classes: "all" for all,
            "present" for classes present in targets,
             or a list of classes to average.
    """
    if probabilities.numel() == 0:
        # only void pixels, the gradients should be 0
        return probabilities * 0.
    C = probabilities.size(1)
    losses = []
    class_to_sum = list(range(C)) if classes in ["all", "present"] else classes
    for c in class_to_sum:
        fg = (targets == c).float()  # foreground for class c
        if classes == "present" and fg.sum() == 0:
            continue
        if C == 1:
            if len(classes) > 1:
                raise ValueError("Sigmoid output possible only with 1 class")
            class_pred = probabilities[:, 0]
        else:
            class_pred = probabilities[:, c]
        errors = (fg - class_pred).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        perm = perm.data
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, _lovasz_grad(fg_sorted)))
    return mean(losses)


def _lovasz_softmax(
    probabilities,
    targets,
    classes="present",
    per_image=False,
    ignore=None
):
    """
    Multi-class Lovasz-Softmax loss

    Args:
        probabilities: [B, C, H, W]
            class probabilities at each prediction (between 0 and 1).
            Interpreted as binary (sigmoid) output
            with outputs of size [B, H, W].
        targets: [B, H, W] ground truth targets (between 0 and C - 1)
        classes: "all" for all,
            "present" for classes present in targets,
            or a list of classes to average.
        per_image: compute the loss per image instead of per batch
        ignore: void class targets
    """
    if per_image:
        loss = mean(
            _lovasz_softmax_flat(
                *_flatten_probabilities(
                    prob.unsqueeze(0),
                    lab.unsqueeze(0),
                    ignore),
                classes=classes)
            for prob, lab in zip(probabilities, targets))
    else:
        loss = _lovasz_softmax_flat(
            *_flatten_probabilities(probabilities, targets, ignore),
            classes=classes)
    return loss


# ------------------------------ CRITERION -------------------------------


class LovaszLossBinary(_Loss):
    def __init__(self, per_image=False, ignore=None):
        super().__init__()
        self.ignore = ignore
        self.per_image = per_image

    def forward(self, logits, targets):
        """
        Args:
            logits: [bs; ...]
            targets: [bs; ...]
        """
        loss = _lovasz_hinge(
            logits,
            targets,
            per_image=self.per_image,
            ignore=self.ignore)
        return loss


class LovaszLossMultiClass(_Loss):
    def __init__(self, per_image=False, ignore=None):
        super().__init__()
        self.ignore = ignore
        self.per_image = per_image

    def forward(self, logits, targets):
        """
        Args:
            logits: [bs; num_classes; ...]
            targets: [bs; ...]
        """
        loss = _lovasz_softmax(
            logits,
            targets,
            per_image=self.per_image,
            ignore=self.ignore)
        return loss


class LovaszLossMultiLabel(_Loss):
    def __init__(self, per_image=False, ignore=None):
        super().__init__()
        self.ignore = ignore
        self.per_image = per_image

    def forward(self, logits, targets):
        """
        Args:
            logits: [bs; num_classes; ...]
            targets: [bs; num_classes; ...]
        """
        losses = [
            _lovasz_hinge(
                logits[:, i, ...],
                targets[:, i, ...],
                per_image=self.per_image,
                ignore=self.ignore)
            for i in range(logits.shape[1])
        ]
        loss = torch.mean(torch.stack(losses))
        return loss


__all__ = ["LovaszLossBinary", "LovaszLossMultiClass", "LovaszLossMultiLabel"]
