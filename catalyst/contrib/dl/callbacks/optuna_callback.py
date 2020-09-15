import optuna

from catalyst.core import Callback, CallbackOrder, IRunner


class OptunaPruningCallback(Callback):
    """
    Optuna callback for pruning unpromising runs

    .. code-block:: python

        import optuna

        from catalyst.dl import SupervisedRunner
        from catalyst.dl.callbacks import OptunaCallback

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
                    OptunaCallback(trial)
                    # some other callbacks ...
                ],
                num_epochs=num_epochs,
            )
            return runner.best_valid_metrics[runner.main_metric]

        study = optuna.create_study()
        study.optimize(objective, n_trials=100, timeout=600)

    Config API is not supported.
    """

    def __init__(self, trial: optuna.Trial = None):
        """
        This callback can be used for early stopping (pruning)
        unpromising runs.

        Args:
             trial: Optuna.Trial for experiment.
        """
        super().__init__(CallbackOrder.External)
        self.trial = trial

    def on_stage_start(self, runner: "IRunner"):
        """
        On stage start hook.
        Takes ``optuna_trial`` from ``Experiment`` for future needs if required.

        Args:
            runner: runner for current experiment
        """
        optuna_trial = getattr(runner.experiment, "optuna_trial", None)
        if self.trial is None and optuna_trial is not None:
            self.trial = optuna_trial

    def on_epoch_end(self, runner: "IRunner"):
        """
        On epoch end hook.

        Considering prune or not to prune current run at current epoch.

        Raises:
            TrialPruned: if current run should be pruned

        Args:
            runner: runner for current experiment
        """
        metric_value = runner.valid_metrics[runner.main_metric]
        self.trial.report(metric_value, step=runner.epoch)
        if self.trial.should_prune():
            message = "Trial was pruned at epoch {}.".format(runner.epoch)
            raise optuna.TrialPruned(message)


OptunaCallback = OptunaPruningCallback

__all__ = ["OptunaCallback", "OptunaPruningCallback"]
