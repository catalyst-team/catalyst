import tqdm
from pprint import pprint
from collections import OrderedDict, defaultdict
from typing import Dict
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

from common.utils.helpers import prepare_model
from common.utils.misc import FrozenClass
from common.dl.callbacks import Callback


class RunnerState(FrozenClass):
    """
    An object that is used to pass internal state during train/valid/infer.
    """

    def __init__(self, **kwargs):
        # data
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
        self.best_metric = None

        # other
        self.is_train = False
        for k, v in kwargs.items():
            setattr(self, k, v)

        self._freeze()


class AbstractModelRunner:
    """
    Abstract model run handler.
    Based on model, it's criterion, optimizer and scheduler stuff.
    """
    def __init__(
            self,
            model: nn.Module,
            criterion: Dict[str, nn.Module] = None,
            optimizer: Dict[str, optim.Optimizer] = None,
            scheduler: Dict[str, optim.lr_scheduler._LRScheduler] = None,
            debug: bool=True):
        """

        :param model: nn.Module instance, your model
        :param criterion: OrderedDict with torch criterions for model training
        :param optimizer: OrderedDict with torch optimizers for model training
        :param scheduler: OrderedDict with torch schedulers for optimizers lrs
        :param debug: boolean flag for all info printing
        """
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
        """
        Inner method for children's classes for model specific initialization.
        As baseline, checks device support and puts model on it.
        :return:
        """
        self.model, self.device = prepare_model(self.model)

    def _init_state(self):
        """
        Inner method for children's classes for state specific initialization.
        :return:
        """
        return RunnerState(device=self.device)

    def run_event(
            self, *,
            callbacks: Dict[str, Callback],
            event: str):
        """
        Innert method to run special event for all available callbacks.

        :param callbacks:
        :param event:
        """
        for callback in callbacks.values():
            getattr(callback, event)(
                state=self.state,
                model=self.model, criterion=self.criterion,
                optimizer=self.optimizer, scheduler=self.scheduler)

    def run(
            self, *,
            loaders: Dict[str, data.DataLoader],
            callbacks: Dict[str, Callback],
            epochs: int = 1,
            mode: str = "train", verbose: bool = False):
        """
        Main method for running train/valid/infer/debug pipeline over model.

        :param loaders: OrderedDict or torch DataLoaders to run on
        :param callbacks: OrderedDict of callback to use
        :param epochs: number of epochs to run
        :param mode: mode - train/infer/debug
        :param verbose: boolean flag for tqdm progress bar
        """
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
                        self.state.output = self.batch_handler(
                            dct=dct, model=self.model, state=self.state)
                    self.run_event(
                        callbacks=callbacks, event="on_batch_end")

                self.run_event(callbacks=callbacks, event="on_loader_end")

            self.run_event(callbacks=callbacks, event="on_epoch_end")

        self.run_event(callbacks=callbacks, event=f"on_{mode}_end")

    def train(
            self, *,
            loaders: Dict[str, data.DataLoader],
            callbacks: Dict[str, Callback],
            epochs: int = 1):
        """
        Main method for training DL models.

        :param loaders: OrderedDict or torch DataLoaders to run on
        :param callbacks: OrderedDict of callback to use
        :param epochs: number of epochs to run
        """
        return self.run(
            loaders=loaders, callbacks=callbacks,
            epochs=epochs, mode="train", verbose=False)

    def infer(self, *,
              loaders: Dict[str, data.DataLoader],
              callbacks: Dict[str, Callback],
              epochs: int = 1):
        """
        Main method for predicting with DL models.

        :param loaders: OrderedDict or torch DataLoaders to run on
        :param callbacks: OrderedDict of callback to use
        :param epochs: number of epochs to run
        """
        return self.run(
            loaders=loaders, callbacks=callbacks,
            epochs=epochs, mode="infer", verbose=True)

    @staticmethod
    def batch_handler(*, dct, model, state):
        raise NotImplementedError
