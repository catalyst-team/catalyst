from typing import List

from catalyst.dl.core import Callback
from .base import BaseExperiment
from catalyst.dl.callbacks import \
    CriterionCallback, OptimizerCallback, SchedulerCallback, \
    CheckpointCallback


class SupervisedExperiment(BaseExperiment):

    def get_callbacks(self, stage: str) -> "List[Callback]":
        callbacks = self._callbacks
        if not stage.startswith("infer"):
            default_callbacks = [
                (self._criterion, CriterionCallback),
                (self._optimizer, OptimizerCallback),
                (self._scheduler, SchedulerCallback),
                ("_default_saver", CheckpointCallback),
            ]

            for key, value in default_callbacks:
                is_already_present = any(
                    isinstance(x, value) for x in callbacks)
                if key is not None and not is_already_present:
                    callbacks.append(value())
        return callbacks


__all__ = ["SupervisedExperiment"]
