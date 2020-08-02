from typing import List

import numpy as np

import torch

from catalyst.core.callbacks import CriterionCallback
from catalyst.core.runner import IRunner


class CutmixCallback(CriterionCallback):
    """
    Callback to do Cutmix augmentation that has been proposed in
    `CutMix: Regularization Strategy to Train Strong Classifiers
    with Localizable Features`_.

    .. warning::
        `catalyst.contrib.dl.callbacks.CutmixCallback` is inherited from
        `catalyst.dl.CriterionCallback` and does its work.
        You may not use them together.

    .. _CutMix\: Regularization Strategy to Train Strong Classifiers with Localizable Features: https://arxiv.org/abs/1905.04899  # noqa: W605, E501, W505
    """

    def __init__(
        self,
        fields: List[str] = ("features",),
        alpha=1.0,
        on_train_only=True,
        **kwargs
    ):
        """
        Args:
            fields (List[str]): list of features which must be affected.
            alpha (float): beta distribution parameter.
            on_train_only (bool): Apply to train only.
                So, if on_train_only is True, use a standard output/metric
                for validation.
        """
        assert (
            len(fields) > 0
        ), "At least one field for CutmixCallback is required"
        assert alpha >= 0, "alpha must be >=0"

        super().__init__(**kwargs)

        self.on_train_only = on_train_only
        self.fields = fields
        self.alpha = alpha
        self.lam = 1
        self.index = None
        self.is_needed = True

    def _compute_loss(self, runner: IRunner, criterion):
        """Computes loss.

        If self.is_needed is ``False`` then calls ``_compute_loss``
        from ``CriterionCallback``, otherwise computes loss value.

        Args:
            runner (IRunner): current runner
            criterion: that is used to compute loss
        """
        if not self.is_needed:
            return super()._compute_loss_value(runner, criterion)

        pred = runner.output[self.output_key]
        y_a = runner.input[self.input_key]
        y_b = runner.input[self.input_key][self.index]
        loss = self.lam * criterion(pred, y_a) + (1 - self.lam) * criterion(
            pred, y_b
        )
        return loss

    def _rand_bbox(self, size, lam):
        """
        Generates top-left and bottom-right coordinates of the box
        of the given size.

        Args:
            size: size of the box
            lam: lambda parameter

        Returns:
            top-left and bottom-right coordinates of the box
        """
        w = size[2]
        h = size[3]
        cut_rat = np.sqrt(1.0 - lam)
        cut_w = np.int(w * cut_rat)
        cut_h = np.int(h * cut_rat)

        cx = np.random.randint(w)
        cy = np.random.randint(h)

        bbx1 = np.clip(cx - cut_w // 2, 0, w)
        bby1 = np.clip(cy - cut_h // 2, 0, h)
        bbx2 = np.clip(cx + cut_w // 2, 0, w)
        bby2 = np.clip(cy + cut_h // 2, 0, h)

        return bbx1, bby1, bbx2, bby2

    def on_loader_start(self, runner: IRunner) -> None:
        """Checks if it is needed for the loader.

        Args:
            runner (IRunner): current runner
        """
        self.is_needed = not self.on_train_only or runner.is_train_loader

    def on_batch_start(self, runner: IRunner) -> None:
        """Mixes data according to Cutmix algorithm.

        Args:
            runner (IRunner): current runner
        """
        if not self.is_needed:
            return

        if self.alpha > 0:
            self.lam = np.random.beta(self.alpha, self.alpha)
        else:
            self.lam = 1

        self.index = torch.randperm(runner.input[self.fields[0]].shape[0])
        self.index.to(runner.device)

        bbx1, bby1, bbx2, bby2 = self._rand_bbox(
            runner.input[self.fields[0]].shape, self.lam
        )

        for f in self.fields:
            runner.input[f][:, :, bbx1:bbx2, bby1:bby2] = runner.input[f][
                self.index, :, bbx1:bbx2, bby1:bby2
            ]

        self.lam = 1 - (
            (bbx2 - bbx1)
            * (bby2 - bby1)
            / (
                runner.input[self.fields[0]].shape[-1]
                * runner.input[self.fields[0]].shape[-2]
            )
        )


__all__ = ["CutmixCallback"]
