from collections import OrderedDict

from catalyst.dl.core import Callback
from .base import BaseExperiment
from catalyst.dl.callbacks import \
    CriterionCallback, OptimizerCallback, SchedulerCallback, \
    CheckpointCallback


class SupervisedExperiment(BaseExperiment):
    def get_callbacks(self, stage: str) -> "OrderedDict[str, Callback]":
        callbacks = OrderedDict(())
        if not stage.startswith("infer"):
            default_callbacks = [
                (self._criterion, "_criterion", CriterionCallback),
                (self._optimizer, "_optimizer", OptimizerCallback),
                (self._scheduler, "_scheduler", SchedulerCallback),
                ("_default_saver", "_saver", CheckpointCallback),
            ]

            for component, callback_name, callback_fn in default_callbacks:
                if component is None:
                    continue

                present_callbacks = [(key, val) for key, val in
                                     callbacks.items() if
                                     isinstance(val, callback_fn)]

                if present_callbacks:
                    for key, val in present_callbacks:
                        callbacks[key] = val
                else:
                    callbacks[callback_name] = callback_fn()
        return callbacks


__all__ = ["SupervisedExperiment"]
