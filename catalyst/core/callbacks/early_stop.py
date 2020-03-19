import os

from catalyst.core import Callback, CallbackNode, CallbackOrder, State

_NUM_BATCH_STEPS = int(os.environ.get("CHECK_BATCH_STEPS", 2))
_NUM_EPOCH_STEPS = int(os.environ.get("CHECK_EPOCH_STEPS", 2))


class CheckRunCallback(Callback):
    def __init__(
        self,
        num_batch_steps: int = _NUM_BATCH_STEPS,
        num_epoch_steps: int = _NUM_EPOCH_STEPS,
    ):
        super().__init__(order=CallbackOrder.External, node=CallbackNode.All)
        self.num_batch_steps = num_batch_steps
        self.num_epoch_steps = num_epoch_steps

    def on_epoch_end(self, state: State):
        if state.epoch >= self.num_epoch_steps:
            state.need_early_stop = True

    def on_batch_end(self, state: State):
        if state.loader_step >= self.num_batch_steps:
            state.need_early_stop = True


class EarlyStoppingCallback(Callback):
    def __init__(
        self,
        patience: int,
        metric: str = "loss",
        minimize: bool = True,
        min_delta: float = 1e-6
    ):
        super().__init__(order=CallbackOrder.External, node=CallbackNode.All)
        self.best_score = None
        self.metric = metric
        self.patience = patience
        self.num_bad_epochs = 0
        self.is_better = None

        if minimize:
            self.is_better = lambda score, best: score <= (best - min_delta)
        else:
            self.is_better = lambda score, best: score >= (best + min_delta)

    def on_epoch_end(self, state: State) -> None:
        if state.stage_name.startswith("infer"):
            return

        score = state.valid_metrics[self.metric]
        if self.best_score is None:
            self.best_score = score
        if self.is_better(score, self.best_score):
            self.num_bad_epochs = 0
            self.best_score = score
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            print(f"Early stop at {state.epoch} epoch")
            state.need_early_stop = True
