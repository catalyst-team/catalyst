from collections import OrderedDict

from torch.optim.lr_scheduler import ReduceLROnPlateau

from catalyst.dl import (
    Callback, CheckpointCallback, CheckRunCallback, ConsoleLogger,
    CriterionCallback, ExceptionCallback, MetricManagerCallback,
    OptimizerCallback, SchedulerCallback, TensorboardLogger, TimerCallback,
    ValidationManagerCallback, VerboseLogger
)
from catalyst.utils.tools.typing import Criterion, Optimizer, Scheduler
from .base import BaseExperiment


class SupervisedExperiment(BaseExperiment):
    """
    Supervised experiment

    The main difference with BaseExperiment that it will
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
            List[Callback]: list of callbacks for experiment
        """
        callbacks = self._callbacks

        default_callbacks = []
        if self._verbose:
            default_callbacks.append(("_verbose", VerboseLogger))
        if self._check_run:
            default_callbacks.append(("_check", CheckRunCallback))

        if not stage.startswith("infer"):
            if self._criterion is not None \
                    and isinstance(self._criterion, Criterion):
                default_callbacks.append(("_criterion", CriterionCallback))
            if self._optimizer is not None \
                    and isinstance(self._optimizer, Optimizer):
                default_callbacks.append(("_optimizer", OptimizerCallback))
            if self._scheduler is not None \
                    and isinstance(
                    self._scheduler, (Scheduler, ReduceLROnPlateau)):
                default_callbacks.append(("_scheduler", SchedulerCallback))

            default_callbacks.append(("_timer", TimerCallback))
            default_callbacks.append(("_metrics", MetricManagerCallback))
            default_callbacks.append(
                ("_validation", ValidationManagerCallback)
            )
            default_callbacks.append(("_saver", CheckpointCallback))
            default_callbacks.append(("_console", ConsoleLogger))
            default_callbacks.append(("_tensorboard", TensorboardLogger))
        default_callbacks.append(("_exception", ExceptionCallback))

        for callback_name, callback_fn in default_callbacks:
            is_already_present = any(
                isinstance(x, callback_fn) for x in callbacks.values()
            )
            if not is_already_present:
                callbacks[callback_name] = callback_fn()
        return callbacks


__all__ = ["SupervisedExperiment"]
