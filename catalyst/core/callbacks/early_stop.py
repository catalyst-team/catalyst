# flake8: noqa
# @TODO: code formatting issue for 20.07 release
from catalyst.core.callback import Callback, CallbackNode, CallbackOrder
from catalyst.core.runner import IRunner


class CheckRunCallback(Callback):
    """Executes only a pipeline part from the ``Experiment``."""

    def __init__(self, num_batch_steps: int = 3, num_epoch_steps: int = 2):
        """
        Args:
            num_batch_steps (int): number of batches to iterate in epoch
            num_epoch_steps (int): number of epoch to perform in a stage
        """
        super().__init__(order=CallbackOrder.external, node=CallbackNode.all)
        self.num_batch_steps = num_batch_steps
        self.num_epoch_steps = num_epoch_steps

    def on_epoch_end(self, runner: IRunner):
        """Check if iterated specified number of epochs.

        Args:
            runner (IRunner): current runner
        """
        if runner.epoch >= self.num_epoch_steps:
            runner.need_early_stop = True

    def on_batch_end(self, runner: IRunner):
        """Check if iterated specified number of batches.

        Args:
            runner (IRunner): current runner
        """
        if runner.loader_batch_step >= self.num_batch_steps:
            runner.need_early_stop = True


class EarlyStoppingCallback(Callback):
    """Early exit based on metric.

    Example of usage in notebook API:

    .. code-block:: python

        runner = SupervisedRunner()
        runner.train(
            ...
            callbacks=[
                ...
                EarlyStoppingCallback(
                    patience=5,
                    metric="my_metric",
                    minimize=True,
                )
                ...
            ]
        )
        ...

    Example of usage in config API:

    .. code-block:: yaml

        stages:
          ...
          stage_N:
            ...
            callbacks_params:
              ...
              early_stopping:
                callback: EarlyStoppingCallback
                # arguments for EarlyStoppingCallback
                patience: 5
                metric: my_metric
                minimize: true
          ...

    """

    def __init__(
        self,
        patience: int,
        metric: str = "loss",
        minimize: bool = True,
        min_delta: float = 1e-6,
    ):
        """
        Args:
            patience (int): number of epochs with no improvement
                after which training will be stopped.
            metric (str): metric name to use for early stopping, default
                is ``"loss"``.
            minimize (bool): if ``True`` then expected that metric should
                decrease and early stopping will be performed only when metric
                stops decreasing. If ``False`` then expected
                that metric should increase. Default value ``True``.
            min_delta (float): minimum change in the monitored metric
                to qualify as an improvement, i.e. an absolute change
                of less than min_delta, will count as no improvement,
                default value is ``1e-6``.
        """
        super().__init__(order=CallbackOrder.external, node=CallbackNode.all)
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
        """Check if should be performed early stopping.

        Args:
            runner (IRunner): current runner
        """
        if runner.stage_name.startswith("infer"):
            return

        score = runner.valid_metrics[self.metric]
        if self.best_score is None or self.is_better(score, self.best_score):
            self.num_bad_epochs = 0
            self.best_score = score
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            print(f"Early stop at {runner.epoch} epoch")
            runner.need_early_stop = True


__all__ = ["CheckRunCallback", "EarlyStoppingCallback"]
