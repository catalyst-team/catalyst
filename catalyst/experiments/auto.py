from collections import OrderedDict

from torch.optim.lr_scheduler import ReduceLROnPlateau

from catalyst.callbacks.criterion import CriterionCallback, ICriterionCallback
from catalyst.callbacks.optimizer import IOptimizerCallback, OptimizerCallback
from catalyst.callbacks.scheduler import ISchedulerCallback, SchedulerCallback
from catalyst.core.callback import Callback
from catalyst.core.functional import check_callback_isinstance
from catalyst.experiments.experiment import Experiment
from catalyst.typing import Criterion, Optimizer, Scheduler


# @TODO: should be mixin-based
class AutoCallbackExperiment(Experiment):
    """
    Auto-optimized experiment.

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
            translates ``runner.*_metrics`` to console and text file.
        TensorboardLogger:
            writes ``runner.*_metrics`` to tensorboard.
        RaiseExceptionCallback:
            will raise exception if needed.
    """

    def get_callbacks(self, stage: str) -> "OrderedDict[str, Callback]":
        """
        Override of ``BaseExperiment.get_callbacks`` method.
        Will add several of callbacks by default in case they missed.

        Args:
            stage: name of stage. It should start with `infer` if you
                don't need default callbacks, as they required only for
                training stages.

        Returns:
            OrderedDict[str, Callback]: Ordered dictionary of callbacks
                for experiment
        """
        callbacks = super().get_callbacks(stage=stage) or OrderedDict()

        # default_callbacks = [(Name, InterfaceClass, InstanceFactory)]
        default_callbacks = []

        if not stage.startswith("infer"):
            if self._criterion is not None and isinstance(self._criterion, Criterion):
                default_callbacks.append(("_criterion", ICriterionCallback, CriterionCallback))
            if self._optimizer is not None and isinstance(self._optimizer, Optimizer):
                default_callbacks.append(("_optimizer", IOptimizerCallback, OptimizerCallback))
            if self._scheduler is not None and isinstance(
                self._scheduler, (Scheduler, ReduceLROnPlateau)
            ):
                default_callbacks.append(("_scheduler", ISchedulerCallback, SchedulerCallback))

        for (callback_name, callback_interface, callback_fn) in default_callbacks:
            callback_interface = callback_interface or callback_fn
            is_already_present = any(
                check_callback_isinstance(x, callback_interface) for x in callbacks.values()
            )
            if not is_already_present:
                callbacks[callback_name] = callback_fn()

        return callbacks


__all__ = ["AutoCallbackExperiment"]
