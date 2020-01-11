from typing import Dict, Optional  # isort:skip
from collections import OrderedDict
from pathlib import Path

import numpy as np

from catalyst.utils.frozen import FrozenClass
from catalyst.utils.metric_manager import MetricManager, TimerManager


class State(FrozenClass):
    def __init__(
        self,
        *,
        logdir: str = None,
        stage: str = "infer",
        num_epochs: int = None,
        main_metric: str = "loss",
        minimize_metric: bool = True,
        valid_loader: str = "train",
        verbose: bool = False,
        checkpoint_data: Dict = None,
        batch_consistant_metrics: bool = True,
        **kwargs
    ):
        self.logdir = Path(logdir) if logdir is not None else None

        # special info
        self.stage = stage
        self.loader_name = None
        self.loaders = None

        # counters
        self.loader_len = 0
        self.batch_size = 0
        self.step = 0
        self.epoch = 0
        self.stage_epoch = 0
        self.num_epochs = num_epochs or np.iinfo(np.int32).max

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

        # extra checkpoint data for saving in checkpoint files
        self.checkpoint_data = checkpoint_data or {}

        # other
        self.need_backward = False
        self.early_stop = False
        for k, v in kwargs.items():
            setattr(self, k, v)

        self.exception: Optional[Exception] = None
        self.need_reraise_exception: bool = True

        self._freeze()

    @property
    def stage_epoch_log(self):
        return self.stage_epoch + 1

    @property
    def epoch_log(self):
        return self.epoch + 1

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
