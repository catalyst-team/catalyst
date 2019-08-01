from collections import OrderedDict

from catalyst.dl.core import Callback
from .base import BaseExperiment
from catalyst.dl.callbacks import \
    CriterionCallback, OptimizerCallback, SchedulerCallback, \
    CheckpointCallback


class SupervisedExperiment(BaseExperiment):
    def get_callbacks(self, stage: str) -> "OrderedDict[str, Callback]":
        callbacks = self._callbacks
        if not stage.startswith("infer"):
            default_callbacks = [
                (self._criterion, "_criterion", CriterionCallback),
                (self._optimizer, "_optimizer", OptimizerCallback),
                (self._scheduler, "_scheduler", SchedulerCallback),
                ("_default_saver", "_saver", CheckpointCallback),
            ]

            for component, callback_name, callback_fn in default_callbacks:
                is_already_present = any(
                    isinstance(x, callback_fn) for x in callbacks.values()
                )
                if component is not None and not is_already_present:
                    callbacks[callback_name] = callback_fn()
        return callbacks


__all__ = ["SupervisedExperiment"]
