from typing import List

import numpy as np

import torch

from catalyst.dl.core.state import RunnerState
from catalyst.dl.callbacks import CriterionCallback


class MixupCallback(CriterionCallback):
    def __init__(self, fields: List[str] = ("features",), alpha=1.0,
                 train_only=True, **kwargs):
        assert len(fields) > 0, \
            "At least one field for MixupCallback is required"
        assert alpha >= 0, "alpha must be>=0"

        super().__init__(**kwargs)

        self.train_only = train_only
        self.fields = fields
        self.alpha = alpha
        self.lam = 1
        self.index = None

    def _is_needed(self, state: RunnerState):
        return not self.train_only or state.loader_name.startswith("train")

    def on_batch_start(self, state: RunnerState):
        if not self._is_needed(state):
            return
        if self.alpha > 0:
            self.lam = np.random.beta(self.alpha, self.alpha)
        else:
            self.lam = 1

        self.index = torch.randperm(state.input[self.fields[0]].shape[0])
        self.index.to(state.device)

        for f in self.fields:
            state.input[f] = self.lam * state.input[f] + \
                (1 - self.lam) * state.input[f][self.index]

    def _compute_loss(self, state: RunnerState, criterion):
        if not self._is_needed(state):
            return super()._compute_loss(state, criterion)
        pred = state.output[self.output_key]
        y_a = state.input[self.input_key]
        y_b = state.input[self.input_key][self.index]

        loss = self.lam * criterion(pred, y_a) + \
            (1 - self.lam) * criterion(pred, y_b)
        return loss


__all__ = ["MixupCallback"]
