import time
from collections import defaultdict
from torchnet import meter
from catalyst.dl.callbacks.utils import get_val_from_metric, \
    process_epoch_metrics
from catalyst.utils.misc import FrozenClass


# TODO Deep refactoring
#  - move metric management to separate class
#  - Remove unused method and params
#  - lr/loss/momentum bypass
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
        optimizer=None,
        scheduler=None,
        stage=None,
        main_metric="loss",
        minimize_metric=True,
        valid_loader="valid",
        reset_step=False,
        mode="infer",
        total_epochs=1,
        **kwargs
    ):
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
        self.is_best_epoch = False
        self.total_epochs = total_epochs

        # metrics
        self.lr = None  # defaultdict(lambda: 0)
        self.momentum = None  # defaultdict(lambda: 0)
        self.loss = None  # defaultdict(lambda: 0)

        self.batch_metrics = defaultdict(lambda: 0)
        self.epoch_metrics = defaultdict(
            lambda: defaultdict(lambda: meter.AverageValueMeter())
        )
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

    @staticmethod
    def on_stage_init_pre(model, stage):
        pass

    @staticmethod
    def on_stage_init_post(model, stage):
        pass

    def on_epoch_start_pre(self):
        pass

    def on_epoch_start_post(self):
        pass

    def on_epoch_end_pre(self):
        if self.mode == "infer":
            return
        best_metrics, valid_metrics, is_best = \
            process_epoch_metrics(
                self.epoch_metrics,
                self.best_metrics,
                valid_loader=self.valid_loader,
                main_metric=self.main_metric,
                minimize=self.minimize_metric)
        valid_metrics = {
            key: value
            for key, value in valid_metrics.items()
            if isinstance(value, float)
        }
        self.best_metrics = {
            key: value
            for key, value in best_metrics.items() if isinstance(value, float)
        }
        self.valid_metrics = valid_metrics
        self.is_best_epoch = is_best

    def on_epoch_end_post(self):
        self.epoch_metrics = defaultdict(
            lambda: defaultdict(lambda: meter.AverageValueMeter())
        )

    def on_loader_start_pre(self):
        pass

    def on_loader_start_post(self):
        self._datatime = time.time()

    def on_loader_end_pre(self):
        lm = self.loader_name
        self.epoch_metrics[lm] = {
            key: get_val_from_metric(value)
            for key, value in self.epoch_metrics[lm].items()
        }

    def on_loader_end_post(self):
        if self.reset_step:
            self.step = None

    def on_batch_start_pre(self):
        self.batch_metrics = defaultdict(lambda: 0)
        self.batch_metrics["base/data_time"] = time.time() - self._datatime

    def on_batch_start_post(self):
        pass

    def on_batch_end_pre(self):
        elapsed_time = time.time() - self._datatime

        self.batch_metrics["base/batch_time"] = elapsed_time
        self.batch_metrics["base/sample_per_second"] = \
            self.batch_size / elapsed_time

    def on_batch_end_post(self):
        lm = self.loader_name
        for key, value in self.batch_metrics.items():
            self.epoch_metrics[lm][key].add(value)
        self.step += self.batch_size
        self._datatime = time.time()
