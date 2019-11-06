from collections import OrderedDict

from catalyst.dl.callbacks import (
    CheckpointCallback, ConsoleLogger, CriterionCallback, OptimizerCallback,
    RaiseExceptionCallback, SchedulerCallback, TensorboardLogger,
    VerboseLogger
)
from catalyst.dl.core import Callback
from .base import BaseExperiment


class SupervisedExperiment(BaseExperiment):
    def get_callbacks(self, stage: str) -> "OrderedDict[str, Callback]":
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
