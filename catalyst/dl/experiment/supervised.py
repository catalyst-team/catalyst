from collections import OrderedDict

from catalyst.dl.callbacks import (
    CheckpointCallback, ConsoleLogger, CriterionCallback, OptimizerCallback,
    RaiseExceptionCallback, SchedulerCallback, TensorboardLogger,
    VerboseLogger
)
from catalyst.dl.core import Callback
from .base import BaseExperiment


class SupervisedExperiment(BaseExperiment):
    """Supervised experiment used mostly in Notebook API

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
            standard Catalyst logger, translates ``state.metrics`` to console
            and text file
        TensorboardLogger:
            will write ``state.metrics`` to tensorboard
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
            default_callbacks.append(("verbose", VerboseLogger))
        if not stage.startswith("infer"):
            default_callbacks.append(("_criterion", CriterionCallback))
            default_callbacks.append(("_optimizer", OptimizerCallback))
            if self._scheduler is not None:
                default_callbacks.append(("_scheduler", SchedulerCallback))
            default_callbacks.append(("_saver", CheckpointCallback))
            default_callbacks.append(("console", ConsoleLogger))
            default_callbacks.append(("tensorboard", TensorboardLogger))
        default_callbacks.append(("exception", RaiseExceptionCallback))

        for callback_name, callback_fn in default_callbacks:
            is_already_present = any(
                isinstance(x, callback_fn) for x in callbacks.values()
            )
            if not is_already_present:
                callbacks[callback_name] = callback_fn()
        return callbacks


__all__ = ["SupervisedExperiment"]
