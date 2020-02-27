from typing import Dict, Optional, Union  # isort:skip
from collections import defaultdict, OrderedDict
from pathlib import Path

import numpy as np

from catalyst.core import Callback
from catalyst.utils.tools.frozen_class import FrozenClass
from catalyst.utils.tools.typing import (
    Criterion, Device, Model, Optimizer, Scheduler, DataLoader
)


STATE_MODEL = Union[Model, OrderedDict[str, Model]]
STATE_CRITERION = Union[Criterion, OrderedDict[str, Criterion]]
STATE_OPTIMIZER = Union[Optimizer, OrderedDict[str, Optimizer]]
STATE_SCHEDULER = Union[Scheduler, OrderedDict[str, Scheduler]]


class _State(FrozenClass):
    def __init__(
        self,
        *,
        device: Device = None,
        model: STATE_MODEL = None,
        criterion: STATE_CRITERION = None,
        optimizer: STATE_OPTIMIZER = None,
        scheduler: STATE_SCHEDULER = None,
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
        self.model: STATE_MODEL = model
        self.criterion: STATE_CRITERION = criterion
        self.optimizer: STATE_OPTIMIZER = optimizer
        self.scheduler: STATE_SCHEDULER = scheduler
        ## extra components - PyTorch device
        self.device: Device = device
        ## extra components - callbacks
        self.callbacks: OrderedDict[str, Callback] = callbacks

        # dataflow - in, out, metrics
        self.batch_in = None
        self.batch_out = None
        ## let's use flatten storage for batch metrics
        ## batch_metrics = {'loss': ..., 'accuracy': ..., 'iou': ...}
        self.batch_metrics = defaultdict(None)
        ## just aggregated (aka mean over all batches)
        ## batch statistics for loader
        ## and global loader metrics, like AUC
        ## loader_metrics = {'loss': ..., 'accuracy': ..., `auc`: ...}
        self.loader_metrics = defaultdict(None)
        ## summarized metrics for different loaders
        ## and global epoch metrics, like lr, momentum
        ## epoch_metrics = {
        ## 'train': {'loss': ...}, 'valid': {'loss': ...},
        ## 'lr': ..., 'momentum': ...,
        ## }
        self.epoch_metrics = defaultdict(None)
        ## #TODO: how to use stage metrics correctly? what for? best selection?
        self.stage_metrics = defaultdict(None)

        # pipeline info
        self.stage_name: str = stage
        self.loader_name: str = None
        self.loader_len: int = 0
        self.batch_size: int = 0
        self.step: int = 0
        self.epoch: int = 0
        self.stage_epoch: int = 0
        self.num_epochs: int = num_epochs or np.iinfo(np.int32).max

        # metrics & validation
        self.main_metric: str = main_metric
        self.minimize_metric: bool = minimize_metric
        self.valid_loader: str = valid_loader

        # logging
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

    def get_attr(self, key, inner_key=None):
        if inner_key is None:
            return getattr(self, key)
        else:
            return getattr(self, key)[inner_key]

    def set_attr(self, value, key, inner_key=None):
        if inner_key is None:
            setattr(self, key, value)
        else:
            getattr(self, key)[inner_key] = value
