import tqdm
from pprint import pprint
from collections import OrderedDict, defaultdict
import torch

from common.utils.helpers import prepare_model
from common.utils.misc import FrozenClass


class RunnerState(FrozenClass):
    """An object that is used to pass internal state during train/valid/infer"""

    def __init__(self, **kwargs):
        # data
        self.input = None
        self.output = None
        self.loader = None
        self.loader_mode = None

        # counters
        self.lr = defaultdict(lambda :0)
        self.momentum = defaultdict(lambda :0)
        self.bs = 0
        self.epoch = 0
        self.iteration = 0
        self.step = 0

        # metrics
        self.loss = None
        self.epoch_metrics = None
        self.best_metrics = None
        self.best_metric = None

        # other
        self.is_train = False
        for k, v in kwargs.items():
            setattr(self, k, v)

        self._freeze()


class AbstractModelRunner:
    def __init__(
            self,
            model, criterion=None, optimizer=None, scheduler=None,
            debug=True):
        assert criterion is None or isinstance(criterion, OrderedDict)
        assert optimizer is None or isinstance(optimizer, OrderedDict)
        assert scheduler is None or isinstance(scheduler, OrderedDict)
        self.model = model
        self.criterion = criterion or {}
        self.optimizer = optimizer or {}
        self.scheduler = scheduler or {}

        if debug:
            pprint(model)
            pprint(criterion)
            pprint(optimizer)
            pprint(scheduler)

        self.state = None
        self._init()

    def _init(self):
        self.model, self.device = prepare_model(self.model)

    def _init_state(self):
        return RunnerState(device=self.device)

    def run_event(self, *, callbacks, event):
        for callback in callbacks.values():
            getattr(callback, event)(
                state=self.state,
                model=self.model, criterion=self.criterion,
                optimizer=self.optimizer, scheduler=self.scheduler)

    def run(
            self, *, loaders, callbacks,
            batch_handler=None, epochs=1,
            mode="train", verbose=False):
        batch_handler = batch_handler or self.batch_handler

        assert isinstance(loaders, OrderedDict)
        assert isinstance(callbacks, OrderedDict)

        self.state = self._init_state()

        self.run_event(callbacks=callbacks, event=f"on_{mode}_start")

        for epoch in range(epochs):
            self.state.epoch = epoch

            self.run_event(callbacks=callbacks, event="on_epoch_start")

            for loader_mode, loader in loaders.items():
                self.state.loader_mode = loader_mode
                self.state.loader = loader

                self.state.is_train = loader_mode.startswith("train")
                self.model.train(self.state.is_train)

                self.run_event(
                    callbacks=callbacks, event="on_loader_start")

                loader = tqdm.tqdm(loader) if verbose else loader
                for i, dct in enumerate(loader):
                    self.state.input = dct

                    self.run_event(
                        callbacks=callbacks, event="on_batch_start")
                    with torch.set_grad_enabled(self.state.is_train):
                        self.state.output = batch_handler(
                            dct=dct, model=self.model, state=self.state)
                    self.run_event(
                        callbacks=callbacks, event="on_batch_end")

                self.run_event(callbacks=callbacks, event="on_loader_end")

            self.run_event(callbacks=callbacks, event="on_epoch_end")

        self.run_event(callbacks=callbacks, event=f"on_{mode}_end")

    def train(self, *, loaders, callbacks, batch_handler=None, epochs=1):
        return self.run(
            loaders=loaders, callbacks=callbacks,
            batch_handler=batch_handler, epochs=epochs,
            mode="train", verbose=False)

    def infer(self, *, loaders, callbacks, batch_handler=None, epochs=1):
        return self.run(
            loaders=loaders, callbacks=callbacks,
            batch_handler=batch_handler, epochs=epochs,
            mode="infer", verbose=True)

    @staticmethod
    def batch_handler(*, dct, model, state):
        raise NotImplementedError
