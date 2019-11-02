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
            default_callbacks.append(
                ("_verbose_logger", "verbose", VerboseLogger)
            )
        if not stage.startswith("infer"):
            default_callbacks.extend([
                (self._criterion, "_criterion", CriterionCallback),
                (self._optimizer, "_optimizer", OptimizerCallback),
                (self._scheduler, "_scheduler", SchedulerCallback),
                ("_default_saver", "_saver", CheckpointCallback),
                ("_console_logger", "console", ConsoleLogger),
                ("_tensorboard_logger", "tensorboard", TensorboardLogger)
            ])
        default_callbacks.append(
            ("_exception", "exception", RaiseExceptionCallback)
        )

        for component, callback_name, callback_fn in default_callbacks:
            is_already_present = any(
                isinstance(x, callback_fn) for x in callbacks.values()
            )
            if component is not None and not is_already_present:
                callbacks[callback_name] = callback_fn()
        return callbacks


__all__ = ["SupervisedExperiment"]
