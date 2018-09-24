from collections import defaultdict
from prometheus.utils.misc import FrozenClass


class RunnerState(FrozenClass):
    """
    An object that is used to pass internal state during train/valid/infer.
    """

    def __init__(self, **kwargs):
        # data
        self.device = None
        self.input = None
        self.output = None
        self.loader = None
        self.loader_mode = None

        # counters
        self.bs = 0
        self.step = 0
        self.epoch = 0

        # metrics
        self.lr = defaultdict(lambda: 0)
        self.momentum = defaultdict(lambda: 0)
        self.loss = None
        self.epoch_metrics = None
        self.best_metrics = None

        # other
        self.is_train = False
        for k, v in kwargs.items():
            setattr(self, k, v)

        self._freeze()
