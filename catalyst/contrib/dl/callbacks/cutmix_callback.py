from typing import List

import numpy as np

import torch

from catalyst.dl import CriterionCallback, State


class CutmixCallback(CriterionCallback):
    """
    Callback to do Cutmix augmentation.

    Paper: https://arxiv.org/pdf/1905.04899.pdf

    Note:
        CutmixCallback is inherited from CriterionCallback and
        does its work.

        You may not use them together.
    """
    def __init__(
        self,
        fields: List[str] = ("features", ),
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
        assert len(fields) > 0, \
            "At least one field for CutmixCallback is required"
        assert alpha >= 0, "alpha must be >=0"

        super().__init__(**kwargs)

        self.on_train_only = on_train_only
        self.fields = fields
        self.alpha = alpha
        self.lam = 1
        self.index = None
        self.is_needed = True

    def _compute_loss(self, state: State, criterion):
        """
        Computes loss.
        If self.is_needed is False then calls _compute_loss
        from CriterionCallback,
        otherwise computes loss value.
        :param state: current state
        :param criterion: that is used to compute loss
        :return: loss value
        """
        if not self.is_needed:
            return super()._compute_loss_value(state, criterion)

        pred = state.output[self.output_key]
        y_a = state.input[self.input_key]
        y_b = state.input[self.input_key][self.index]
        loss = self.lam * criterion(pred, y_a) + \
            (1 - self.lam) * criterion(pred, y_b)
        return loss

    def _rand_bbox(self, size, lam):
        """
        Generates top-left and bottom-right coordinates of the box
        of the given size.
        :param size: size of the box
        :param lam: lambda parameter
        :return: top-left and bottom-right coordinates of the box
        """
        w = size[2]
        h = size[3]
        cut_rat = np.sqrt(1. - lam)
        cut_w = np.int(w * cut_rat)
        cut_h = np.int(h * cut_rat)

        cx = np.random.randint(w)
        cy = np.random.randint(h)

        bbx1 = np.clip(cx - cut_w // 2, 0, w)
        bby1 = np.clip(cy - cut_h // 2, 0, h)
        bbx2 = np.clip(cx + cut_w // 2, 0, w)
        bby2 = np.clip(cy + cut_h // 2, 0, h)

        return bbx1, bby1, bbx2, bby2

    def on_loader_start(self, state: State):
        """
        Checks if it is needed for the loader.
        :param state: current state
        :return: void
        """
        self.is_needed = not self.on_train_only or \
            state.loader_name.startswith("train")

    def on_batch_start(self, state: State):
        """
        Mixes data according to Cutmix algorithm.
        :param state: current state
        :return: void
        """
        if not self.is_needed:
            return

        if self.alpha > 0:
            self.lam = np.random.beta(self.alpha, self.alpha)
        else:
            self.lam = 1

        self.index = torch.randperm(state.input[self.fields[0]].shape[0])
        self.index.to(state.device)

        bbx1, bby1, bbx2, bby2 = \
            self._rand_bbox(state.input[self.fields[0]].shape, self.lam)

        for f in self.fields:
            state.input[f][:, :, bbx1:bbx2, bby1:bby2] = \
                state.input[f][self.index, :, bbx1:bbx2, bby1:bby2]

        self.lam = 1 - (
            (bbx2 - bbx1) * (bby2 - bby1) / (
                state.input[self.fields[0]].shape[-1] *
                state.input[self.fields[0]].shape[-2]
            )
        )
