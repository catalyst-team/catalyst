from typing import List, Optional, Tuple, TYPE_CHECKING, Union
from abc import ABC, abstractmethod

import torch

from catalyst.contrib.schedulers import BatchScheduler, OneCycleLRWithWarmup
from catalyst.core.callback import ISchedulerCallback
from catalyst.typing import TorchOptimizer
from catalyst.utils.misc import get_attr
from catalyst.utils.torch import get_optimizer_momentum, get_optimizer_momentum_list

if TYPE_CHECKING:
    from catalyst.core.runner import IRunner


class SchedulerCallback(ISchedulerCallback):
    """Scheduler callback, abstraction over scheduler step.

    Args:
        scheduler_key: scheduler name, if ``None``,
            default is ``None``.
        mode: scheduler mode, should be one of
            ``"epoch"`` or ``"batch"``, default is ``None``.
            If ``None`` and object is instance of ``BatchScheduler``
            or ``OneCycleLRWithWarmup`` then will be used ``"batch"``
            otherwise - ``"epoch"``.
        loader_key: loader name to look after for ReduceLROnPlateau scheduler
        metric_key: metric name to forward to scheduler
            object, if ``None`` then will be used main metric
            specified in experiment.

    .. note::
        Please follow the `minimal examples`_ sections for more use cases.

        .. _`minimal examples`: https://github.com/catalyst-team/catalyst#minimal-examples  # noqa: E501, W505
    """

    def __init__(
        self,
        scheduler_key: str = None,
        mode: str = None,
        loader_key: str = None,
        metric_key: str = None,
    ):
        """Init."""
        super().__init__()
        if loader_key is not None or metric_key is not None:
            assert loader_key is not None and metric_key is not None, (
                "For metric reduction `SchedulerCallback` "
                "requires both `loader_key` and `metric_key` specified."
            )
            self._use_metric_reduction = True
        else:
            self._use_metric_reduction = False
        assert mode in ("batch", "epoch", None)
        self.scheduler_key = scheduler_key
        self.mode = mode
        self.loader_key = loader_key
        self.metric_key = metric_key
        self.scheduler = None

    @staticmethod
    def _scheduler_step(scheduler, reduced_metric=None):
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(reduced_metric)
        else:
            scheduler.step()

        lr_list = [param_group["lr"] for param_group in scheduler.optimizer.param_groups]
        momentum_list = get_optimizer_momentum_list(scheduler.optimizer)
        return lr_list, momentum_list

    def _update_lr_and_momentum_in_metrics_dict(
        self,
        metrics_dict: dict,
        lr_list: List[float],
        momentum_list: List[Union[float, None]],
    ):
        """Update learning rate and momentum in metrics_dict
        (consider only 0-th param group)

        Args:
            metrics_dict: batch_metrics or epoch_metrics
            lr_list: lr for each param group
            momentum_list: momentum for each param group

        """
        lr = lr_list[0]
        momentum = momentum_list[0]

        lr_key = f"lr/{self.scheduler_key}" if self.scheduler_key is not None else "lr"
        metrics_dict[lr_key] = lr

        if momentum is not None:
            momentum_key = (
                f"momentum/{self.scheduler_key}"
                if self.scheduler_key is not None
                else "momentum"
            )
            metrics_dict[momentum_key] = momentum

    def make_batch_step(self, runner: "IRunner") -> None:
        """Perform scheduler step and update batch metrics

        Args:
            runner: current runner
        """
        lr_list, momentum_list = self._scheduler_step(scheduler=self.scheduler)
        self._update_lr_and_momentum_in_metrics_dict(
            runner.batch_metrics, lr_list, momentum_list
        )

    def make_epoch_step(self, runner: "IRunner") -> None:
        """Perform scheduler step and update epoch metrics

        Args:
            runner: current runner
        """
        if self._use_metric_reduction:
            reduced_metric = runner.epoch_metrics[self.loader_key][self.metric_key]
        else:
            reduced_metric = None
        lr_list, momentum_list = self._scheduler_step(
            scheduler=self.scheduler, reduced_metric=reduced_metric
        )
        # @TODO: remove trick to save pure epoch-based metrics, like lr/momentum
        self._update_lr_and_momentum_in_metrics_dict(
            runner.epoch_metrics["_epoch_"], lr_list, momentum_list
        )

    def on_experiment_start(self, runner: "IRunner") -> None:
        """Event handler."""
        self.scheduler = get_attr(runner, key="scheduler", inner_key=self.scheduler_key)
        assert self.scheduler is not None

        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            assert self.loader_key is not None and self.metric_key is not None, (
                "For `ReduceLROnPlateau` scheduler `SchedulerCallback` "
                "required both `loader_key` and `metric_key` specified"
            )

        if self.mode is None:
            if isinstance(self.scheduler, BatchScheduler):
                self.mode = "batch"
            else:
                self.mode = "epoch"

        if isinstance(self.scheduler, OneCycleLRWithWarmup) and self.mode == "batch":
            self.scheduler.reset()
        assert self.mode is not None

    def on_loader_start(self, runner: "IRunner") -> None:
        """Event handler."""
        if (
            runner.is_train_loader
            and isinstance(self.scheduler, OneCycleLRWithWarmup)
            and self.mode == "batch"
        ):
            self.scheduler.recalculate(
                loader_batch_len=runner.loader_batch_len,
                current_batch_step=runner.batch_step,
            )

    def on_batch_end(self, runner: "IRunner") -> None:
        """Event handler."""
        if runner.is_train_loader and self.mode == "batch":
            self.make_batch_step(runner=runner)

    def on_epoch_end(self, runner: "IRunner") -> None:
        """Event handler."""
        if self.mode == "epoch":
            self.make_epoch_step(runner=runner)


class ILRUpdater(ABC, ISchedulerCallback):
    """Class interface for all Lr updaters."""

    def __init__(self, optimizer_key: str = None):
        """
        Args:
            optimizer_key: which optimizer key to use
                for learning rate scheduling
        """
        super().__init__()
        self.init_lr = 0
        self.optimizer_key = optimizer_key
        self.optimizer = None

    @abstractmethod
    def calc_lr(self) -> float:
        """Interface for calculating learning rate."""
        pass

    @abstractmethod
    def calc_momentum(self) -> float:
        """Interface for calculating momentum"""
        pass

    @staticmethod
    def _update_lr(optimizer: TorchOptimizer, new_lr: float) -> None:
        for pg in optimizer.param_groups:
            pg["lr"] = new_lr

    @staticmethod
    def _update_momentum(optimizer: TorchOptimizer, new_momentum: float) -> None:
        if "betas" in optimizer.param_groups[0]:
            for pg in optimizer.param_groups:
                pg["betas"] = (new_momentum, pg["betas"][1])
        else:
            for pg in optimizer.param_groups:
                pg["momentum"] = new_momentum

    def _update_optimizer(self, optimizer: TorchOptimizer) -> Tuple[float, float]:
        new_lr = self.calc_lr()
        if new_lr is not None:
            self._update_lr(optimizer, new_lr)
        else:
            new_lr = optimizer.param_groups[0]["lr"]

        new_momentum = self.calc_momentum()
        if new_momentum is not None:
            self._update_momentum(optimizer, new_momentum)
        else:
            new_momentum = get_optimizer_momentum(optimizer)

        return new_lr, new_momentum

    def update_optimizer(self, runner: "IRunner") -> None:
        """Update learning rate and momentum in runner.

        Args:
            runner: current runner
        """
        lr, momentum = self._update_optimizer(optimizer=self.optimizer)

        if self.optimizer_key is not None:
            runner.batch_metrics[f"lr_{self.optimizer_key}"] = lr
            runner.batch_metrics[f"momentum_{self.optimizer_key}"] = momentum
        else:
            runner.batch_metrics["lr"] = lr
            runner.batch_metrics["momentum"] = momentum

    def on_experiment_start(self, runner: "IRunner") -> None:
        """Event handler."""
        self.optimizer = optimizer = get_attr(
            runner, key="optimizer", inner_key=self.optimizer_key
        )
        self.optimizer = optimizer
        self.init_lr = optimizer.param_groups[0]["lr"]

    def on_loader_start(self, runner: "IRunner") -> None:
        """Event handler."""
        if runner.is_train_loader:
            self.update_optimizer(runner=runner)

    def on_batch_end(self, runner: "IRunner") -> None:
        """Event handler."""
        if runner.is_train_loader:
            self.update_optimizer(runner=runner)


class LRFinder(ILRUpdater):
    """
    Helps you find an optimal learning rate for a model, as per suggestion of
    `Cyclical Learning Rates for Training Neural Networks`_ paper.
    Learning rate is increased in linear or log scale, depending on user input.

    See `How Do You Find A Good Learning Rate`_ article for details.

    .. _Cyclical Learning Rates for Training Neural Networks:
        https://arxiv.org/abs/1506.01186
    .. _How Do You Find A Good Learning Rate:
        https://sgugger.github.io/how-do-you-find-a-good-learning-rate.html
    """

    def __init__(
        self,
        final_lr: float,
        scale: str = "log",
        num_steps: Optional[int] = None,
        optimizer_key: str = None,
    ):
        """
        Args:
            final_lr: final learning rate to try with
            scale: learning rate increasing scale ("log" or "linear")
            num_steps:  number of batches to try, if None - whole loader would be used.
            optimizer_key: which optimizer key to use for learning rate scheduling

        Raises:
            NotImplementedError: if invalid scale value.
        """
        super().__init__(optimizer_key=optimizer_key)

        self.final_lr = final_lr
        self.scale = scale
        self.num_steps = num_steps
        self.multiplier = 0
        self.lr_step = 0
        self.iteration = 0

        self._calc_lr = None
        if scale == "log":
            self._calc_lr = self._calc_lr_log
        elif scale == "linear":
            self._calc_lr = self._calc_lr_linear
        else:
            raise NotImplementedError("Not supported")

    def _calc_lr_log(self):
        return self.init_lr * self.multiplier ** self.iteration

    def _calc_lr_linear(self):
        return self.init_lr + self.lr_step * self.iteration

    def calc_lr(self):
        """Calculates learning rate.

        Returns:
            learning rate.
        """
        res = self._calc_lr()
        self.iteration += 1
        return res

    def calc_momentum(self):
        """Calculates new momentum."""
        pass

    def on_loader_start(self, runner: "IRunner"):
        """Loader start hook. Updates scheduler statistics.

        Args:
            runner: current runner
        """
        if runner.is_train_loader:
            lr_step = self.final_lr / self.init_lr
            self.num_steps = self.num_steps or runner.loader_batch_len
            self.multiplier = lr_step ** (1 / self.num_steps)
            self.lr_step = (self.final_lr - self.init_lr) / self.num_steps

        super().on_loader_start(runner=runner)

    def on_batch_end(self, runner: "IRunner"):
        """Batch end hook. Make scheduler step and stops iterating if needed.

        Args:
            runner: current runner

        Raises:
            KeyboardInterrupt: at the end of LRFinder
        """
        super().on_batch_end(runner=runner)
        if self.iteration > self.num_steps:
            # runner.need_early_stop = True
            raise KeyboardInterrupt("End of LRFinder")


__all__ = ["ISchedulerCallback", "SchedulerCallback", "ILRUpdater", "LRFinder"]
