from collections import OrderedDict

from torch.optim.lr_scheduler import ReduceLROnPlateau

from catalyst.dl import (
    Callback,
    CriterionCallback,
    OptimizerCallback,
    SchedulerCallback,
)
from catalyst.utils.tools.typing import Criterion, Optimizer, Scheduler

from .core import Experiment


class SupervisedExperiment(Experiment):
    """
    Supervised experiment.

    The main difference with Experiment that it will
    add several callbacks by default if you haven't.

    Here are list of callbacks by default:
        CriterionCallback:
            measures loss with specified ``criterion``.
        OptimizerCallback:
            abstraction over ``optimizer`` step.
        SchedulerCallback:
            only in case if you provided scheduler to your experiment does
            `lr_scheduler.step`
        CheckpointCallback:
            saves model and optimizer state each epoch callback to save/restore
            your model/criterion/optimizer/metrics.
        ConsoleLogger:
            standard Catalyst logger,
            translates ``state.*_metrics`` to console and text file
        TensorboardLogger:
            will write ``state.*_metrics`` to tensorboard
        RaiseExceptionCallback:
            will raise exception if needed
    """

    def get_callbacks(self, stage: str) -> "OrderedDict[str, Callback]":
        """
        Override of ``BaseExperiment.get_callbacks`` method.
        Will add several of callbacks by default in case they missed.

        Args:
            stage (str): name of stage. It should start with `infer` if you
                don't need default callbacks, as they required only for
                training stages.

        Returns:
            (OrderedDict[str, Callback]): Ordered dictionary of callbacks
                for experiment
        """
        callbacks = super().get_callbacks(stage=stage) or OrderedDict()

        default_callbacks = []

        if not stage.startswith("infer"):
            if self._criterion is not None and isinstance(
                self._criterion, Criterion
            ):
                default_callbacks.append(("_criterion", CriterionCallback))
            if self._optimizer is not None and isinstance(
                self._optimizer, Optimizer
            ):
                default_callbacks.append(("_optimizer", OptimizerCallback))
            if self._scheduler is not None and isinstance(
                self._scheduler, (Scheduler, ReduceLROnPlateau)
            ):
                default_callbacks.append(("_scheduler", SchedulerCallback))

        for callback_name, callback_fn in default_callbacks:
            is_already_present = any(
                isinstance(x, callback_fn) for x in callbacks.values()
            )
            if not is_already_present:
                callbacks[callback_name] = callback_fn()

        return callbacks


__all__ = ["SupervisedExperiment"]
