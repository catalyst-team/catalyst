from collections import OrderedDict

from torch.optim.lr_scheduler import ReduceLROnPlateau

from catalyst.dl import (
    AMPOptimizerCallback,
    Callback,
    CriterionCallback,
    IOptimizerCallback,
    ISchedulerCallback,
    OptimizerCallback,
    SchedulerCallback,
)
from catalyst.dl.experiment.experiment import Experiment
from catalyst.dl.utils import check_amp_available, check_callback_isinstance
from catalyst.tools.typing import Criterion, Optimizer, Scheduler


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
            stage (str): name of stage. It should start with `infer` if you
                don't need default callbacks, as they required only for
                training stages.

        Returns:
            OrderedDict[str, Callback]: Ordered dictionary of callbacks
                for experiment
        """
        callbacks = super().get_callbacks(stage=stage) or OrderedDict()

        # default_callbacks = [(Name, InterfaceClass, InstanceFactory)]
        default_callbacks = []

        is_amp_enabled = (
            self.distributed_params.get("amp", False) and check_amp_available()
        )
        optimizer_cls = (
            AMPOptimizerCallback if is_amp_enabled else OptimizerCallback
        )

        if not stage.startswith("infer"):
            if self._criterion is not None and isinstance(
                self._criterion, Criterion
            ):
                default_callbacks.append(
                    ("_criterion", None, CriterionCallback)
                )
            if self._optimizer is not None and isinstance(
                self._optimizer, Optimizer
            ):
                default_callbacks.append(
                    ("_optimizer", IOptimizerCallback, optimizer_cls)
                )
            if self._scheduler is not None and isinstance(
                self._scheduler, (Scheduler, ReduceLROnPlateau)
            ):
                default_callbacks.append(
                    ("_scheduler", ISchedulerCallback, SchedulerCallback)
                )

        for (
            callback_name,
            callback_interface,
            callback_fn,
        ) in default_callbacks:
            callback_interface = callback_interface or callback_fn
            is_already_present = any(
                check_callback_isinstance(x, callback_interface)
                for x in callbacks.values()
            )
            if not is_already_present:
                callbacks[callback_name] = callback_fn()

        return callbacks


__all__ = ["SupervisedExperiment"]
