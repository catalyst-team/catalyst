from typing import Dict, Optional  # isort:skip
from collections import defaultdict, OrderedDict
from pathlib import Path
from copy import deepcopy

import numpy as np

from catalyst.core import Callback
from catalyst.utils.tools.frozen_class import FrozenClass
from catalyst.utils.tools.typing import (
    Criterion, Device, Model, Optimizer, Scheduler, DataLoader
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
        is_check_run: bool = False,
        **kwargs,
    ):
        # main part
        ## data
        self.loaders: OrderedDict[str, DataLoader] = None
        ## components
        self.model: Model = model
        self.criterion: Criterion = criterion
        self.optimizer: Optimizer = optimizer
        self.scheduler: Scheduler = scheduler
        ## extra components - PyTorch device
        self.device: Device = device
        ## extra components - callbacks
        self.callbacks: OrderedDict[str, Callback] = callbacks

        # dataflow - in, out, metrics
        self.batch_in = None
        self.batch_out = None
        ## let's use flatten storage for metrics
        defaulf_metrics = {
            "loss": None,
            "lr": None,
            "momentum": None,
            "data_time": None,
            "model_time": None,
            "batch_time": None,
        }
        self.batch_metrics = deepcopy(defaulf_metrics)
        self.loader_metrics = deepcopy(defaulf_metrics)
        self.epoch_metrics = deepcopy(defaulf_metrics)
        self.stage_metrics = deepcopy(defaulf_metrics)

        # pipeline info
        self.stage_name: str = stage
        self.loader_name: str = None
        self.loader_len: int = 0
        self.batch_size: int = 0
        self.step: int = 0
        self.epoch: int = 0
        self.stage_epoch: int = 0
        self.num_epochs: int = num_epochs or np.iinfo(np.int32).max

        # metrics & logging
        self.main_metric: str = main_metric
        self.minimize_metric: bool = minimize_metric
        self.valid_loader: str = valid_loader
        self.logdir: Path = Path(logdir) if logdir is not None else None
        # extra checkpoint data for saving in checkpoint files
        self.checkpoint_data: Dict = checkpoint_data or {}

        # other
        self.is_check_run: bool = is_check_run
        self.need_backward_pass: bool = False
        self.need_early_stop: bool = False
        self.need_exception_reraise: bool = True
        self.exception: Optional[Exception] = None

        # kwargs
        for k, v in kwargs.items():
            setattr(self, k, v)

        self._freeze()

    @property
    def stage_epoch_log(self):
        return self.stage_epoch + 1

    @property
    def epoch_log(self):
        return self.epoch + 1
