from collections import defaultdict
from torchnet import meter
from prometheus.utils.misc import FrozenClass


class RunnerState(FrozenClass):
    """
    An object that is used to pass internal state during train/valid/infer.
    """

    def __init__(self, **kwargs):
        # special info
        self.device = None
        self.logdir = None
        self.loader = None
        self.loader_mode = None

        # data pipeline
        self.input = None
        self.output = None

        # counters
        self.step = 0
        self.epoch = 0

        # metrics
        self.lr = defaultdict(lambda: 0)
        self.momentum = defaultdict(lambda: 0)
        self.loss = defaultdict(lambda: 0)

        self.batch_metrics = defaultdict(lambda: 0)
        self.epoch_metrics = defaultdict(
            lambda: defaultdict(lambda: meter.AverageValueMeter()))
        self.valid_metrics = None
        self.best_metrics = None

        # other
        self.is_train = False
        for k, v in kwargs.items():
            setattr(self, k, v)

        self._freeze()
