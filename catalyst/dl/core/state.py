from typing import Dict, Optional  # isort:skip
from collections import defaultdict, OrderedDict
from pathlib import Path

from torch.optim.optimizer import Optimizer

from catalyst.utils.frozen import FrozenClass
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
        logdir: str = None,
        stage: str = "infer",
        num_epochs: int = 1,
        main_metric: str = "loss",
        minimize_metric: bool = True,
        valid_loader: str = "valid",
        verbose: bool = False,
        checkpoint_data: Dict = None,
        batch_consistant_metrics: bool = True,
        **kwargs
    ):
        self.logdir = Path(logdir) if logdir is not None else None
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler

        # special info
        self.stage = stage
        self.device = device
        self.loader_name = None
        self.phase = None

        # data pipeline
        self.input = None
        self.output = None

        # counters
        self.loader_len = 0
        self.batch_size = 0
        self.step = 0
        self.epoch = 0
        self.stage_epoch = 0
        self.num_epochs = num_epochs

        # metrics & logging
        self.main_metric = main_metric
        self.minimize_metric = minimize_metric
        self.valid_loader = valid_loader
        self.metrics = MetricManager(
            valid_loader=valid_loader,
            main_metric=main_metric,
            minimize=minimize_metric,
            batch_consistant_metrics=batch_consistant_metrics
        )
        self.verbose: bool = verbose
        self.loggers = OrderedDict()
        self.timer = TimerManager()

        # base metrics
        single_optimizer = isinstance(optimizer, Optimizer)
        self.lr = None if single_optimizer else defaultdict(lambda: None)
        self.momentum = None if single_optimizer else defaultdict(lambda: None)
        self.loss = None

        # extra checkpoint data for saving in checkpoint files
        self.checkpoint_data = checkpoint_data or {}

        # other
        self.need_backward = False
        self.early_stop = False
        self.model_grads = {}
        for k, v in kwargs.items():
            setattr(self, k, v)

        self.exception: Optional[Exception] = None
        self.need_reraise_exception: bool = True

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
        values = {}
        for key, value in zip(
            ["_base/lr", "_base/momentum"], [self.lr, self.momentum]
        ):
            if value is not None:
                if isinstance(value, dict):
                    for k, v in value.items():
                        values[f"{key}/{k}"] = v
                else:
                    values[key] = value

        values.update(self.timer.elapsed)

        values["_timers/_fps"] = \
            self.batch_size / self.timer.elapsed["_timers/batch_time"]

        self.metrics.add_batch_value(metrics_dict=values)

    def on_stage_start_pre(self):
        pass

    def on_stage_start_post(self):
        pass

    def on_stage_end_pre(self):
        pass

    def on_stage_end_post(self):
        pass

    def on_epoch_start_pre(self):
        self.metrics.begin_epoch()
        pass

    def on_epoch_start_post(self):
        pass

    def on_epoch_end_pre(self):
        if not self.stage.startswith("infer"):
            self.metrics.end_epoch_train()

    def on_epoch_end_post(self):
        pass

    def on_loader_start_pre(self):
        self.metrics.begin_loader(self.loader_name)

    def on_loader_start_post(self):
        pass

    def on_loader_end_pre(self):
        self.metrics.end_loader()

    def on_loader_end_post(self):
        pass

    def on_batch_start_pre(self):
        self.metrics.begin_batch()

    def on_batch_start_post(self):
        pass

    def on_batch_end_pre(self):
        pass

    def on_batch_end_post(self):
        self._handle_runner_metrics()
        self.metrics.end_batch()

    def on_exception_pre(self):
        pass

    def on_exception_post(self):
        pass

    @property
    def stage_epoch_log(self):
        return self.stage_epoch + 1

    @property
    def epoch_log(self):
        return self.epoch + 1


__all__ = ["RunnerState"]
