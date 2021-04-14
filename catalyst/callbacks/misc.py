from abc import ABC, abstractmethod

# from tqdm.auto import tqdm
from torch.utils.model_zoo import tqdm

from catalyst.core.callback import Callback, CallbackNode, CallbackOrder
from catalyst.core.runner import IRunner
from catalyst.tools.metric_handler import MetricHandler
from catalyst.tools.time_manager import TimeManager
from catalyst.utils.misc import is_exception

EPS = 1e-8


class IBatchMetricHandlerCallback(ABC, Callback):
    """@TODO: docs"""

    def __init__(self, metric_key: str, minimize: bool = True, min_delta: float = 1e-6):
        """@TODO: docs"""
        super().__init__(order=CallbackOrder.external, node=CallbackNode.all)
        self.is_better = MetricHandler(minimize=minimize, min_delta=min_delta)
        self.metric_key = metric_key
        self.best_score = None

    @abstractmethod
    def handle_score_is_better(self, runner: "IRunner"):
        """Event handler."""
        pass

    @abstractmethod
    def handle_score_is_not_better(self, runner: "IRunner"):
        """Event handler."""
        pass

    def on_loader_start(self, runner: "IRunner") -> None:
        """Event handler."""
        self.best_score = None

    def on_batch_end(self, runner: "IRunner") -> None:
        """Event handler."""
        score = runner.batch_metrics[self.metric_key]
        if self.best_score is None or self.is_better(score, self.best_score):
            self.best_score = score
            self.handle_score_is_better(runner=runner)
        else:
            self.handle_score_is_not_better(runner=runner)


class IEpochMetricHandlerCallback(ABC, Callback):
    """@TODO: docs"""

    def __init__(
        self, loader_key: str, metric_key: str, minimize: bool = True, min_delta: float = 1e-6,
    ):
        """@TODO: docs"""
        super().__init__(order=CallbackOrder.external, node=CallbackNode.all)
        self.is_better = MetricHandler(minimize=minimize, min_delta=min_delta)
        self.loader_key = loader_key
        self.metric_key = metric_key
        self.best_score = None

    @abstractmethod
    def handle_score_is_better(self, runner: "IRunner"):
        """Event handler."""
        pass

    @abstractmethod
    def handle_score_is_not_better(self, runner: "IRunner"):
        """Event handler."""
        pass

    def on_stage_start(self, runner: "IRunner") -> None:
        """Event handler."""
        self.best_score = None

    def on_epoch_end(self, runner: "IRunner") -> None:
        """Event handler."""
        score = runner.epoch_metrics[self.loader_key][self.metric_key]
        if self.best_score is None or self.is_better(score, self.best_score):
            self.best_score = score
            self.handle_score_is_better(runner=runner)
        else:
            self.handle_score_is_not_better(runner=runner)


class EarlyStoppingCallback(IEpochMetricHandlerCallback):
    """Stage early stop based on metric.

    Args:
        patience: number of epochs with no improvement
            after which training will be stopped.
        loader_key: loader key for early stopping (based on metric score over the dataset)
        metric_key: metric key for early stopping (based on metric score over the dataset)
        minimize: if ``True`` then expected that metric should
            decrease and early stopping will be performed only when metric
            stops decreasing. If ``False`` then expected
            that metric should increase. Default value ``True``.
        min_delta: minimum change in the monitored metric
            to qualify as an improvement, i.e. an absolute change
            of less than min_delta, will count as no improvement,
            default value is ``1e-6``.
    """

    def __init__(
        self,
        patience: int,
        loader_key: str,
        metric_key: str,
        minimize: bool,
        min_delta: float = 1e-6,
    ):
        """Init."""
        super().__init__(
            loader_key=loader_key, metric_key=metric_key, minimize=minimize, min_delta=min_delta,
        )
        self.patience = patience
        self.num_no_improvement_epochs = 0

    def handle_score_is_better(self, runner: "IRunner"):
        """Event handler."""
        self.num_no_improvement_epochs = 0

    def handle_score_is_not_better(self, runner: "IRunner"):
        """Event handler."""
        self.num_no_improvement_epochs += 1
        if self.num_no_improvement_epochs >= self.patience:
            # print(f"Early stop at {runner.epoch} epoch")
            runner.need_early_stop = True


class TimerCallback(Callback):
    """Logs pipeline execution time."""

    def __init__(self):
        """Initialisation for TimerCallback."""
        super().__init__(order=CallbackOrder.metric + 1, node=CallbackNode.all)
        self.timer = TimeManager()

    def on_loader_start(self, runner: "IRunner") -> None:
        """Loader start hook.

        Args:
            runner: current runner
        """
        self.timer.reset()
        self.timer.start("_timer/batch_time")
        self.timer.start("_timer/data_time")

    def on_loader_end(self, runner: "IRunner") -> None:
        """Loader end hook.

        Args:
            runner: current runner
        """
        self.timer.reset()

    def on_batch_start(self, runner: "IRunner") -> None:
        """Batch start hook.

        Args:
            runner: current runner
        """
        self.timer.stop("_timer/data_time")
        self.timer.start("_timer/model_time")

    def on_batch_end(self, runner: "IRunner") -> None:
        """Batch end hook.

        Args:
            runner: current runner
        """
        self.timer.stop("_timer/model_time")
        self.timer.stop("_timer/batch_time")

        # @TODO: just a trick
        self.timer.elapsed["_timer/_fps"] = runner.batch_size / (
            self.timer.elapsed["_timer/batch_time"] + EPS
        )
        for key, value in self.timer.elapsed.items():
            runner.batch_metrics[key] = value

        self.timer.reset()
        self.timer.start("_timer/batch_time")
        self.timer.start("_timer/data_time")


class TqdmCallback(Callback):
    """Logs the params into tqdm console."""

    def __init__(self):
        super().__init__(order=CallbackOrder.external, node=CallbackNode.master)
        self.tqdm: tqdm = None
        self.step = 0

    def on_loader_start(self, runner: "IRunner"):
        """Init tqdm progress bar."""
        self.step = 0
        self.tqdm = tqdm(
            total=runner.loader_batch_len,
            desc=f"{runner.stage_epoch_step}/{runner.stage_epoch_len}"
            f" * Epoch ({runner.loader_key})",
            # leave=True,
            # ncols=0,
            # file=sys.stdout,
        )

    def on_batch_end(self, runner: "IRunner"):
        """Update tqdm progress bar at the end of each batch."""
        batch_metrics = {k: float(v) for k, v in runner.batch_metrics.items()}
        self.tqdm.set_postfix(
            **{
                k: "{:3.3f}".format(v) if v > 1e-3 else "{:1.3e}".format(v)
                for k, v in sorted(batch_metrics.items())
            }
        )
        self.tqdm.update()

    def on_loader_end(self, runner: "IRunner"):
        """Cleanup and close tqdm progress bar."""
        # self.tqdm.visible = False
        # self.tqdm.leave = True
        # self.tqdm.disable = True
        self.tqdm.clear()
        self.tqdm.close()
        self.tqdm = None
        self.step = 0

    def on_exception(self, runner: "IRunner"):
        """Called if an Exception was raised."""
        exception = runner.exception
        if not is_exception(exception):
            return

        if isinstance(exception, KeyboardInterrupt):
            if self.tqdm is not None:
                self.tqdm.write("Keyboard Interrupt")
                self.tqdm.clear()
                self.tqdm.close()
                self.tqdm = None


class CheckRunCallback(Callback):
    """Executes only a pipeline part from the run.

    Args:
        num_batch_steps: number of batches to iterate in epoch
        num_epoch_steps: number of epoch to perform in a stage

    Minimal working example (Notebook API):

    .. code-block:: python

        import torch
        from torch.utils.data import DataLoader, TensorDataset
        from catalyst import dl

        # data
        num_samples, num_features = int(1e4), int(1e1)
        X, y = torch.rand(num_samples, num_features), torch.rand(num_samples)
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=32, num_workers=1)
        loaders = {"train": loader, "valid": loader}

        # model, criterion, optimizer, scheduler
        model = torch.nn.Linear(num_features, 1)
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters())
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [3, 6])

        # model training
        runner = dl.SupervisedRunner()
        runner.train(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            loaders=loaders,
            logdir="./logdir",
            num_epochs=8,
            verbose=True,
            callbacks=[
                dl.CheckRunCallback(num_batch_steps=3, num_epoch_steps=3)
            ]
        )

    """

    def __init__(self, num_batch_steps: int = 3, num_epoch_steps: int = 3):
        """Init."""
        super().__init__(order=CallbackOrder.external, node=CallbackNode.all)
        self.num_batch_steps = num_batch_steps
        self.num_epoch_steps = num_epoch_steps

    def on_epoch_end(self, runner: "IRunner"):
        """Check if iterated specified number of epochs.

        Args:
            runner: current runner
        """
        if runner.stage_epoch_step >= self.num_epoch_steps:
            runner.need_early_stop = True

    def on_batch_end(self, runner: "IRunner"):
        """Check if iterated specified number of batches.

        Args:
            runner: current runner
        """
        if runner.loader_batch_step >= self.num_batch_steps:
            runner.need_early_stop = True


__all__ = [
    "TimerCallback",
    "TqdmCallback",
    "CheckRunCallback",
    "IBatchMetricHandlerCallback",
    "IEpochMetricHandlerCallback",
    "EarlyStoppingCallback",
]
