from abc import ABC, abstractmethod

from tqdm.auto import tqdm

from catalyst.core.callback import Callback, CallbackOrder
from catalyst.core.runner import IRunner
from catalyst.extras.metric_handler import MetricHandler
from catalyst.extras.time_manager import TimeManager

EPS = 1e-8


class IEpochMetricHandlerCallback(ABC, Callback):
    """Docs"""

    def __init__(
        self,
        loader_key: str,
        metric_key: str,
        minimize: bool = True,
        min_delta: float = 1e-6,
    ):
        """Docs"""
        super().__init__(order=CallbackOrder.external)
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

    def on_experiment_start(self, runner: "IRunner") -> None:
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
    """Early stop based on metric.

    Args:
        patience: number of epochs with no improvement
            after which training will be stopped.
        loader_key: loader key for early stopping
            (based on metric score over the dataset)
        metric_key: metric key for early stopping
            (based on metric score over the dataset)
        minimize: if ``True`` then expected that metric should
            decrease and early stopping will be performed only when metric
            stops decreasing. If ``False`` then expected
            that metric should increase. Default value ``True``.
        min_delta: minimum change in the monitored metric
            to qualify as an improvement, i.e. an absolute change
            of less than min_delta, will count as no improvement,
            default value is ``1e-6``.

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
            num_epochs=100,
            callbacks=[
                dl.EarlyStoppingCallback(
                    loader_key="valid",
                    metric_key="loss",
                    minimize=True,
                    patience=3,
                    min_delta=1e-2
                )
            ]
        )

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
            loader_key=loader_key,
            metric_key=metric_key,
            minimize=minimize,
            min_delta=min_delta,
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
            runner.need_early_stop = True


class TimerCallback(Callback):
    """Logs pipeline execution time.

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
            num_epochs=1,
            verbose=True,
            callbacks=[dl.TimerCallback()]
        )

    You should see additional extra metrics, such as:

    - ``_timer/_fps`` - number handled samples per second during run.
    - ``_timer/batch_time`` - time required for single batch handling.
    - ``_timer/data_time`` - time required for single batch data preparation handling.
    - ``_timer/model_time`` - time required for single batch model forwarding.

    Moreover, you could use it throught ``timeit=True`` flag:

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
            num_epochs=1,
            verbose=True,
            timeit=True,
        )

    """

    def __init__(self):
        """Init."""
        super().__init__(order=CallbackOrder.metric + 1)
        self.timer = TimeManager()

    def on_loader_start(self, runner: "IRunner") -> None:
        """Event handler."""
        self.timer.reset()
        self.timer.start("_timer/batch_time")
        self.timer.start("_timer/data_time")

    def on_loader_end(self, runner: "IRunner") -> None:
        """Event handler."""
        self.timer.reset()

    def on_batch_start(self, runner: "IRunner") -> None:
        """Event handler."""
        self.timer.stop("_timer/data_time")
        self.timer.start("_timer/model_time")

    def on_batch_end(self, runner: "IRunner") -> None:
        """Event handler."""
        self.timer.stop("_timer/model_time")
        self.timer.stop("_timer/batch_time")

        self.timer.elapsed["_timer/_fps"] = runner.batch_size / (
            self.timer.elapsed["_timer/batch_time"] + EPS
        )
        for key, value in self.timer.elapsed.items():
            runner.batch_metrics[key] = value

        self.timer.reset()
        self.timer.start("_timer/batch_time")
        self.timer.start("_timer/data_time")


class TqdmCallback(Callback):
    """Logs the params into tqdm console.

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
            num_epochs=1,
            callbacks=[dl.TqdmCallback()]
        )

    You should see a tqdm progress bar during the training.

    Moreover, you could use it throught ``verbose=True`` flag:

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
            num_epochs=1,
            verbose=True,
        )

    """

    def __init__(self):
        super().__init__(order=CallbackOrder.external)
        self.tqdm: tqdm = None
        self.step = 0

    def on_loader_start(self, runner: "IRunner"):
        """Init tqdm progress bar."""
        if runner.engine.process_index > 0:
            return
        self.step = 0
        self.tqdm = tqdm(
            total=runner.loader_batch_len,
            desc=f"{runner.epoch_step}/{runner.num_epochs}"
            f" * Epoch ({runner.loader_key})",
        )

    def on_batch_end(self, runner: "IRunner"):
        """Update tqdm progress bar at the end of each batch."""
        if runner.engine.process_index > 0:
            return
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
        if runner.engine.process_index > 0:
            return
        # self.tqdm.visible = False
        # self.tqdm.leave = True
        # self.tqdm.disable = True
        self.tqdm.clear()
        self.tqdm.close()
        self.tqdm = None
        self.step = 0

    def on_exception(self, runner: "IRunner"):
        """Called if an Exception was raised."""
        if runner.engine.process_index > 0:
            return
        ex = runner.exception
        if not ((ex is not None) and isinstance(ex, BaseException)):
            return

        if isinstance(ex, KeyboardInterrupt):
            if self.tqdm is not None:
                self.tqdm.write("Keyboard Interrupt")
                self.tqdm.clear()
                self.tqdm.close()
                self.tqdm = None


class CheckRunCallback(Callback):
    """Executes only a pipeline part from the run.

    Args:
        num_batch_steps: number of batches to iterate in epoch
        num_epoch_steps: number of epoch to perform in an experiment

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
        super().__init__(order=CallbackOrder.external)
        self.num_batch_steps = num_batch_steps
        self.num_epoch_steps = num_epoch_steps

    def on_batch_end(self, runner: "IRunner"):
        """Check if iterated specified number of batches.

        Args:
            runner: current runner
        """
        if runner.loader_batch_step >= self.num_batch_steps:
            runner.need_early_stop = True

    def on_epoch_end(self, runner: "IRunner"):
        """Check if iterated specified number of epochs.

        Args:
            runner: current runner
        """
        if runner.epoch_step >= self.num_epoch_steps:
            runner.need_early_stop = True


__all__ = [
    "CheckRunCallback",
    "EarlyStoppingCallback",
    "TimerCallback",
    "TqdmCallback",
]
