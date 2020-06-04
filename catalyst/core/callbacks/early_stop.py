from catalyst.core.callback import Callback, CallbackNode, CallbackOrder
from catalyst.core.runner import IRunner


class CheckRunCallback(Callback):
    """@TODO: Docs. Contribution is welcome."""

    def __init__(self, num_batch_steps: int = 3, num_epoch_steps: int = 2):
        """@TODO: Docs. Contribution is welcome."""
        super().__init__(order=CallbackOrder.External, node=CallbackNode.All)
        self.num_batch_steps = num_batch_steps
        self.num_epoch_steps = num_epoch_steps

    def on_epoch_end(self, runner: IRunner):
        """@TODO: Docs. Contribution is welcome."""
        if runner.epoch >= self.num_epoch_steps:
            runner.need_early_stop = True

    def on_batch_end(self, runner: IRunner):
        """@TODO: Docs. Contribution is welcome."""
        if runner.loader_batch_step >= self.num_batch_steps:
            runner.need_early_stop = True


class EarlyStoppingCallback(Callback):
    """@TODO: Docs. Contribution is welcome."""

    def __init__(
        self,
        patience: int,
        metric: str = "loss",
        minimize: bool = True,
        min_delta: float = 1e-6,
    ):
        """@TODO: Docs. Contribution is welcome."""
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

    def on_epoch_end(self, runner: IRunner) -> None:
        """@TODO: Docs. Contribution is welcome."""
        if runner.stage_name.startswith("infer"):
            return

        score = runner.valid_metrics[self.metric]
        if self.best_score is None:
            self.best_score = score
        if self.is_better(score, self.best_score):
            self.num_bad_epochs = 0
            self.best_score = score
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            print(f"Early stop at {runner.epoch} epoch")
            runner.need_early_stop = True
