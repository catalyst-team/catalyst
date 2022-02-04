from typing import TYPE_CHECKING

from catalyst.core.callback import Callback, CallbackOrder
from catalyst.extras.metric_handler import MetricHandler
from catalyst.settings import SETTINGS

if SETTINGS.optuna_required:
    import optuna

if TYPE_CHECKING:
    from catalyst.core.runner import IRunner


class OptunaPruningCallback(Callback):
    """Optuna callback for pruning unpromising runs.
    This callback can be used for early stopping (pruning) unpromising runs.

    Args:
        trial:  Optuna.Trial for the experiment.
        loader_key: loader key for best model selection
            (based on metric score over the dataset)
        metric_key: metric key for best model selection
            (based on metric score over the dataset)
        minimize: boolean flag to minimize the required metric
        min_delta: minimal delta for metric improve

    .. code-block:: python

        import optuna

        from catalyst.dl import SupervisedRunner, OptunaPruningCallback

        # some python code ...

        def objective(trial: optuna.Trial):
            # standard optuna code for model and/or optimizer suggestion ...
            runner = SupervisedRunner()
            runner.train(
                model=model,
                loaders=loaders,
                criterion=criterion,
                optimizer=optimizer,
                callbacks=[
                    OptunaPruningCallback(trial)
                    # some other callbacks ...
                ],
                num_epochs=num_epochs,
            )
            return runner.best_valid_metrics[runner.valid_metric]

        study = optuna.create_study()
        study.optimize(objective, n_trials=100, timeout=600)
    """

    def __init__(
        self,
        trial: "optuna.Trial",
        loader_key: str,
        metric_key: str,
        minimize: bool,
        min_delta: float = 1e-6,
    ):
        """Init."""
        super().__init__(CallbackOrder.External)
        self.trial = trial
        self.loader_key = loader_key
        self.metric_key = metric_key
        self.minimize = minimize
        self.is_better = MetricHandler(minimize=minimize, min_delta=min_delta)
        self.best_score = None

    def on_epoch_end(self, runner: "IRunner"):
        """Considering prune or not to prune current run at the end of the epoch.

        Args:
            runner: runner for current experiment

        Raises:
            TrialPruned: if current run should be pruned
        """
        score = runner.epoch_metrics[self.loader_key][self.metric_key]
        if self.best_score is None or self.is_better(score, self.best_score):
            self.best_score = score
        self.trial.report(score, step=runner.epoch_step)

        # @TODO: hack
        self.trial.best_score = self.best_score

        if self.trial.should_prune():
            message = "Trial was pruned at epoch {}.".format(runner.epoch_step)
            raise optuna.TrialPruned(message)


__all__ = ["OptunaPruningCallback"]
