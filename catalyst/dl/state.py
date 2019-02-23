import time
from collections import defaultdict

from pathlib import Path
from torchnet import meter
from torch.optim.optimizer import Optimizer

from catalyst.utils.misc import FrozenClass
from .metric_manager import MetricManager


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
        stage=None,
        main_metric="valid/loss",
        minimize_metric=True,
        valid_loader="valid",
        reset_step=False,
        mode="infer",
        total_epochs=1,
        logdir='logs',
        **kwargs
    ):
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
        self.reset_step = reset_step

        self.main_metric = main_metric
        self.minimize_metric = minimize_metric
        self.valid_loader = valid_loader

        # data pipeline
        self.input = None
        self.output = None

        # counters
        self._datatime = time.time()
        self.loader_len = 0
        self.batch_size = 0
        self.step = 0
        self.epoch = 0
        self.total_epochs = total_epochs

        self.metrics = MetricManager(main_metric, minimize_metric)

        # metrics
        self.lr = None
        self.momentum = None
        self.loss = None

        self.valid_metrics = None
        self.best_metrics = None

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
