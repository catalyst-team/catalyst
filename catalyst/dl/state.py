import time
from collections import defaultdict
from torchnet import meter
from catalyst.dl.callbacks.utils import get_val_from_metric, \
    process_epoch_metrics
from catalyst.utils.misc import FrozenClass


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
        **kwargs
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler

        # special info
        self.stage = stage
        self.mode = "infer"
        self.device = device
        self.loader_mode = None
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
        self.key2device = defaultdict(lambda: True)
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

    @staticmethod
    def on_train_start_pre(state):
        pass

    @staticmethod
    def on_train_start_post(state):
        pass

    @staticmethod
    def on_train_end_pre(state):
        pass

    @staticmethod
    def on_train_end_post(state):
        pass

    @staticmethod
    def on_infer_start_pre(state):
        pass

    @staticmethod
    def on_infer_start_post(state):
        pass

    @staticmethod
    def on_infer_end_pre(state):
        pass

    @staticmethod
    def on_infer_end_post(state):
        pass

    @staticmethod
    def on_epoch_start_pre(state):
        pass

    @staticmethod
    def on_epoch_start_post(state):
        pass

    @staticmethod
    def on_epoch_end_pre(state):
        if state.mode == "infer":
            return

        best_metrics, valid_metrics, is_best = \
            process_epoch_metrics(
                state.epoch_metrics,
                state.best_metrics,
                valid_loader=state.valid_loader,
                main_metric=state.main_metric,
                minimize=state.minimize_metric)
        valid_metrics = {
            key: value
            for key, value in valid_metrics.items()
            if isinstance(value, float)
        }
        state.best_metrics = {
            key: value
            for key, value in best_metrics.items() if isinstance(value, float)
        }
        state.valid_metrics = valid_metrics
        state.is_best_epoch = is_best

    @staticmethod
    def on_epoch_end_post(state):
        state.epoch_metrics = defaultdict(
            lambda: defaultdict(lambda: meter.AverageValueMeter())
        )

    @staticmethod
    def on_loader_start_pre(state):
        pass

    @staticmethod
    def on_loader_start_post(state):
        state._datatime = time.time()

    @staticmethod
    def on_loader_end_pre(state):
        lm = state.loader_mode
        state.epoch_metrics[lm] = {
            key: get_val_from_metric(value)
            for key, value in state.epoch_metrics[lm].items()
        }

    @staticmethod
    def on_loader_end_post(state):
        if state.reset_step:
            state.step = None

    @staticmethod
    def on_batch_start_pre(state):
        state.batch_metrics = defaultdict(lambda: 0)
        state.batch_metrics["base/data_time"] = time.time() - state._datatime

    @staticmethod
    def on_batch_start_post(state):
        pass

    @staticmethod
    def on_batch_end_pre(state):
        elapsed_time = time.time() - state._datatime

        state.batch_metrics["base/batch_time"] = elapsed_time
        state.batch_metrics["base/sample_per_second"] = \
            state.batch_size / elapsed_time

    @staticmethod
    def on_batch_end_post(state):
        lm = state.loader_mode
        for key, value in state.batch_metrics.items():
            state.epoch_metrics[lm][key].add(value)
        state.step += state.batch_size
        state._datatime = time.time()
