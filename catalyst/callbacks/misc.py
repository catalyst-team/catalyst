from abc import ABC, abstractmethod

from tqdm.auto import tqdm

from catalyst.core.callback import Callback, CallbackNode, CallbackOrder
from catalyst.core.runner import IRunner
from catalyst.tools.time_manager import TimeManager

EPS = 1e-8


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


class VerboseCallback(Callback):
    """Logs the params into console."""

    def __init__(self):
        super().__init__(order=CallbackOrder.logging, node=CallbackNode.master)
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
        self.tqdm.set_postfix(
            **{
                k: "{:3.3f}".format(v) if v > 1e-3 else "{:1.3e}".format(v)
                for k, v in sorted(runner.batch_metrics.items())
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

    # def on_exception(self, runner: "IRunner"):
    #     """Called if an Exception was raised."""
    #     exception = runner.exception
    #     if not is_exception(exception):
    #         return
    #
    #     if isinstance(exception, KeyboardInterrupt):
    #         if self.tqdm is not None:
    #             self.tqdm.write("Early exiting")
    #         runner.need_exception_reraise = False


class CheckRunCallback(Callback):
    """Executes only a pipeline part from the ``Experiment``.

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
        """
        Args:
            num_batch_steps: number of batches to iterate in epoch
            num_epoch_steps: number of epoch to perform in a stage
        """
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


class IRunnerMetricHandler(ABC):
    def __init__(self, minimize: bool = True, min_delta: float = 1e-6):
        super().__init__()
        self.minimize = minimize
        self.best_score = None

        if minimize:
            self.is_better = lambda score, best: score <= (best - min_delta)
        else:
            self.is_better = lambda score, best: score >= (best + min_delta)

    @abstractmethod
    def handle_improvement(self, runner: "IRunner"):
        pass

    @abstractmethod
    def handle_no_improvement(self, runner: "IRunner"):
        pass


class IBatchMetricHandlerCallback(Callback, IRunnerMetricHandler):
    def __init__(
        self, metric_key: str, minimize: bool = True, min_delta: float = 1e-6,
    ):
        Callback.__init__(self, order=CallbackOrder.external, node=CallbackNode.all)
        IRunnerMetricHandler.__init__(self, minimize=minimize, min_delta=min_delta)
        self.metric_key = metric_key

    def on_loader_start(self, runner: "IRunner") -> None:
        self.best_score = None

    def on_batch_end(self, runner: "IRunner") -> None:
        score = runner.batch_metrics[self.metric_key]
        if self.best_score is None or self.is_better(score, self.best_score):
            self.best_score = score
            self.handle_improvement(runner=runner)
        else:
            self.handle_no_improvement(runner=runner)


class IEpochMetricHandlerCallback(Callback, IRunnerMetricHandler):
    def __init__(
        self, loader_key: str, metric_key: str, minimize: bool = True, min_delta: float = 1e-6,
    ):
        Callback.__init__(self, order=CallbackOrder.external, node=CallbackNode.all)
        IRunnerMetricHandler.__init__(self, minimize=minimize, min_delta=min_delta)
        self.loader_key = loader_key
        self.metric_key = metric_key

    def on_epoch_end(self, runner: "IRunner") -> None:
        score = runner.epoch_metrics[self.loader_key][self.metric_key]
        if self.best_score is None or self.is_better(score, self.best_score):
            self.best_score = score
            self.handle_improvement(runner=runner)
        else:
            self.handle_no_improvement(runner=runner)


class EarlyStoppingCallback(IEpochMetricHandlerCallback):
    """Stage early stop based on metric."""

    def __init__(
        self,
        patience: int,
        loader_key: str,
        metric_key: str,
        minimize: bool = True,
        min_delta: float = 1e-6,
    ):
        """
        Args:
            patience: number of epochs with no improvement
                after which training will be stopped.
            metric_key: metric name to use for early stopping.
            minimize: if ``True`` then expected that metric should
                decrease and early stopping will be performed only when metric
                stops decreasing. If ``False`` then expected
                that metric should increase. Default value ``True``.
            min_delta: minimum change in the monitored metric
                to qualify as an improvement, i.e. an absolute change
                of less than min_delta, will count as no improvement,
                default value is ``1e-6``.
        """
        super().__init__(
            loader_key=loader_key, metric_key=metric_key, minimize=minimize, min_delta=min_delta,
        )
        self.patience = patience
        self.num_no_improvement_epochs = 0

    def handle_improvement(self, runner: "IRunner"):
        self.num_no_improvement_epochs = 0

    def handle_no_improvement(self, runner: "IRunner"):
        self.num_no_improvement_epochs += 1
        if self.num_no_improvement_epochs >= self.patience:
            # print(f"Early stop at {runner.epoch} epoch")
            runner.need_early_stop = True


class TopNEpochMetricHandlerCallback(IEpochMetricHandlerCallback):
    def __init__(
        self,
        loader_key: str,
        metric_key: str,
        minimize: bool = True,
        min_delta: float = 1e-6,
        save_n_best: int = 1,
    ):
        super().__init__(
            loader_key=loader_key, metric_key=metric_key, minimize=minimize, min_delta=min_delta,
        )
        self.save_n_best = save_n_best
        self.top_best_metrics = []

    def handle_improvement(self, runner: "IRunner"):
        self.top_best_metrics.append((self.best_score, runner.stage_epoch_step,))

        self.top_best_metrics = sorted(
            self.top_best_metrics, key=lambda x: x[0], reverse=not self.minimize,
        )
        if len(self.top_best_metrics) > self.save_n_best:
            self.top_best_metrics.pop(-1)

    def handle_no_improvement(self, runner: "IRunner"):
        pass

    def on_stage_end(self, runner: "IRunner") -> None:
        log_message = "Top-N best epochs:\n"
        log_message += "\n".join(
            [
                "{epoch}\t{metric:3.4f}".format(epoch=epoch, metric=metric)
                for metric, epoch in self.top_best_metrics
            ]
        )
        print(log_message)


__all__ = [
    "TimerCallback",
    "VerboseCallback",
    "CheckRunCallback",
    "IRunnerMetricHandler",
    "IBatchMetricHandlerCallback",
    "IEpochMetricHandlerCallback",
    "EarlyStoppingCallback",
    "TopNEpochMetricHandlerCallback",
]
