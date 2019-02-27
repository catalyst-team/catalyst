from pathlib import Path
from torch.optim.optimizer import Optimizer

from catalyst.utils.misc import FrozenClass
from .metric_manager import MetricManager, TimerManager


# TODO Deep refactoring
#  - lr/loss/momentum bypass (how to deal when multiple optimizers?)
class RunnerState(FrozenClass):
    """
    An object that is used to pass internal state during train/valid/infer.
    """

    def __init__(
        self,
        *,
        device=None,
        model=None,
        criterion=None,
        optimizer: Optimizer = None,
        scheduler=None,
        mode="infer",
        stage=None,
        main_metric="loss",
        minimize_metric=True,
        valid_loader="valid",
        total_epochs=1,
        logdir="logs",
        verbose=False,
        **kwargs
    ):
        # @TODO: refactor
        # hack to prevent cycle imports
        from .callbacks.loggers import (
            VerboseLogger, ConsoleLogger, TensorboardLogger)

        self.logdir = Path(logdir)
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler

        # special info
        self.stage = stage
        self.mode = mode
        self.device = device
        self.loader_name = None

        # data pipeline
        self.input = None
        self.output = None

        # counters
        self.loader_len = 0
        self.batch_size = 0
        self.step = 0
        self.epoch = 0
        self.total_epochs = total_epochs

        # metrics & logging
        self.main_metric = main_metric
        self.minimize_metric = minimize_metric
        self.valid_loader = valid_loader
        self.metrics = MetricManager(
            valid_loader=valid_loader,
            main_metric=main_metric,
            minimize=minimize_metric
        )
        self.loggers = [ConsoleLogger(), TensorboardLogger()]
        if verbose:
            self.loggers.insert(0, VerboseLogger())
        self.timer = TimerManager()

        # base metrics
        self.lr = None
        self.momentum = None
        self.loss = None

        # other
        self.is_train = False
        for k, v in kwargs.items():
            setattr(self, k, v)

        self._freeze()

    def get_key(self, key, inner_key=None):
        if inner_key is None:
            return getattr(self, key)
        else:
            return getattr(self, key)[inner_key]

    def set_key(self, value, key, inner_key=None):
        if inner_key is None:
            setattr(self, key, value)
        else:
            getattr(self, key)[inner_key] = value

    def _handle_runner_metrics(self):
        values = {
            "base/lr": self.lr,
            "base/momentum": self.momentum,
            "loss": self.loss
        }

        values.update(self.timer.elapsed)

        values["_fps"] = \
            self.batch_size / self.timer.elapsed["base/batch_time"]

        self.metrics.add_batch_value(metrics_dict=values)

    def on_stage_start_pre(self):
        for logger in self.loggers:
            logger.on_stage_start(self)

    def on_stage_end_post(self):
        for logger in self.loggers:
            logger.on_stage_end(self)

    def on_epoch_start_pre(self):
        for logger in self.loggers:
            logger.on_epoch_start(self)

    def on_epoch_end_post(self):
        for logger in self.loggers:
            logger.on_epoch_end(self)

    def on_loader_start_pre(self):
        self.metrics.begin_loader(self.loader_name)
        for logger in self.loggers:
            logger.on_loader_start(self)

    def on_loader_end_post(self):
        self.metrics.end_loader()
        for logger in self.loggers:
            logger.on_loader_end(self)

    def on_batch_start_pre(self):
        self.metrics.begin_batch()

    def on_batch_end_post(self):
        self._handle_runner_metrics()
        self.metrics.end_batch()
        for logger in self.loggers:
            logger.on_batch_end(self)
