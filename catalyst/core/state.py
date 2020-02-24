from typing import Dict, Optional  # isort:skip
from collections import defaultdict, OrderedDict
from pathlib import Path

import numpy as np

from catalyst.core import Callback
from catalyst.utils.tools.frozen_class import FrozenClass
from catalyst.utils.tools.typing import (
    Criterion, Device, Model, Optimizer, Scheduler
)


class _State(FrozenClass):
    def __init__(
        self,
        *,
        device: Device = None,
        model: Model = None,
        criterion: Criterion = None,
        optimizer: Optimizer = None,
        scheduler: Scheduler = None,
        callbacks: OrderedDict[str, Callback]= None,
        logdir: str = None,
        stage: str = "infer",
        num_epochs: int = None,
        main_metric: str = "loss",
        minimize_metric: bool = True,
        valid_loader: str = "train",
        checkpoint_data: Dict = None,
        **kwargs,
    ):
        # main part
        self.loaders = None
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.callbacks = callbacks

        # main metrics
        single_optimizer = isinstance(optimizer, Optimizer)
        single_criterion = isinstance(criterion, Criterion)
        self.batch_metrics = {
            "loss": None if single_criterion else defaultdict(lambda: None),
            "lr": None if single_optimizer else defaultdict(lambda: None),
            "momentum": None if single_optimizer else defaultdict(lambda: None),
            "data_time": None,
            "model_time": None,
            "batch_time": None,
        }

        # data pipeline
        self.input = None
        self.output = None

        # logging
        self.logdir = Path(logdir) if logdir is not None else None

        # special info
        self.stage = stage
        self.loader_name = None
        self.loader = None

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

        # extra checkpoint data for saving in checkpoint files
        self.checkpoint_data = checkpoint_data or {}

        # other
        self.need_backward = False
        self.early_stop = False
        for k, v in kwargs.items():
            setattr(self, k, v)

        self.exception: Optional[Exception] = None
        self.need_exception_reraise: bool = True

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
