from typing import List
from catalyst.dl.core.state import RunnerState
from catalyst.dl.callbacks import CriterionCallback
import numpy as np
import torch


class MixupCallback(CriterionCallback):
    def __init__(self, fields: List[str] = ('features',), alpha=1.0, train_only=True, **kwargs):
        assert alpha >= 0, 'alpha must be>=0'

        super(MixupCallback, self).__init__(**kwargs)

        self.train_only = train_only
        self.fields = fields
        self.alpha = alpha
        self.lam = 1
        self.index = None

    def on_batch_start(self, state: RunnerState):
        if self.train_only and state.loader_name != 'train':
            return
        if self.alpha > 0:
            self.lam = np.random.beta(self.alpha, self.alpha)
        else:
            self.lam = 1

        self.index = torch.randperm(state.input[self.fields[0]].shape[0])
        if state.input[self.fields[0]].is_cuda:
            self.index = self.index.cuda()

        for f in self.fields:
            state.input[f] = self.lam * state.input[f] + (1 - self.lam) * state.input[f][self.index]

    def _compute_loss(self, state: RunnerState, criterion):
        if self.train_only and state.loader_name != 'train':
            return super(MixupCallback, self)._compute_loss(state, criterion)
        pred = state.output[self.output_key]
        y_a = state.input[self.input_key]
        y_b = state.input[self.input_key][self.index]

        return self.lam * criterion(pred, y_a) + (1 - self.lam) * criterion(pred, y_b)


__all__ = ['MixupCallback']
