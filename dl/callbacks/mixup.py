from typing import List
from catalyst.data.functional import compute_mixup_lambda, mixup_torch
from .core import Callback


class MixupCallback(Callback):
    def __init__(
        self, mixup_keys: List[str], alpha: float, share_lambda: bool = True
    ):
        self.mixup_keys = mixup_keys
        self.alpha = alpha
        self.share_lambda = share_lambda

    def on_batch_start(self, state):
        lambda_ = compute_mixup_lambda(
            state.batch_size, self.alpha, self.share_lambda
        )
        for key in self.mixup_keys:
            state.input[key] = mixup_torch(state.input[key], lambda_=lambda_)
