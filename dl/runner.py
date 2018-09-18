import tqdm
from pprint import pprint
from collections import OrderedDict, defaultdict
from argparse import Namespace
from typing import Dict
from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

from common.utils.factory import UtilsFactory
from common.utils.misc import FrozenClass
from common.dl.callbacks import Callback


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

        self.stage = None
        self.state = None
        self._init()

    def _init(self):
        """
        Inner method for children's classes for model specific initialization.
        As baseline, checks device support and puts model on it.
        :return:
        """
        self.model, self.device = UtilsFactory.prepare_model(self.model)

    def _init_state(self, *, mode, **kwargs) -> RunnerState:
        """
        Inner method for children's classes for state specific initialization.
        :return: RunnerState with all necessary parameters.
        """
        additional_kwargs = {}
        # transfer previous counters from old state
        if self.state is not None:
            additional_kwargs = {
                "step": self.state.step,
                "epoch": self.state.epoch,
                "best_metrics": self.state.best_metrics
            }
        return RunnerState(
            device=self.device, model=self.model, stage=self.stage,
            _criterion=self.criterion,
            _optimizer=self.optimizer,
            _scheduler=self.scheduler,
            **kwargs,
            **additional_kwargs)

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
            getattr(callback, event)(state=self.state)

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

        self.state = self._init_state(mode=mode)

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

    def train_stage(
            self, *,
            loaders: Dict[str, data.DataLoader],
            callbacks: Dict[str, Callback],
            epochs: int = 1, verbose: bool = False):
        """
        One stage training method.

        :param loaders: OrderedDict or torch DataLoaders to run on
        :param callbacks: OrderedDict of callback to use
        :param epochs: number of epochs to run
        :param verbose: verbose flag
        """
        self.run(
            loaders=loaders, callbacks=callbacks,
            epochs=epochs, mode="train", verbose=verbose)

    def train(
            self, *,
            loaders: Dict[str, data.DataLoader],
            args,
            stages_config: Dict[str, Dict] = None,
            verbose: bool = False):
        """
        Main method for training DL models.

        :param loaders: OrderedDict or torch DataLoaders to run on
        :param args: console args
        :param stages_config: config
        :param verbose: verbose flag
        """
        loggers = UtilsFactory.create_loggers(args.logdir, loaders)

        for key, value in stages_config.items():
            self.stage = key
            UtilsFactory.prepare_stage_args(args=args, stage_config=value)
            pprint(args)

            callbacks = self.prepare_callbacks(
                loggers=loggers, callbacks_params=value["callbacks_params"],
                args=args, mode="train", stage=key)
            pprint(callbacks)

            self.run_event(callbacks=callbacks, event="on_stage_init")
            self.criterion, self.optimizer, self.scheduler = \
                UtilsFactory.prepare_stage_stuff(
                    model=self.model, stage_config=value)

            self.run_event(callbacks=callbacks, event="on_stage_start")
            self.train_stage(
                loaders=loaders, callbacks=callbacks,
                epochs=args.epochs, verbose=verbose)
            self.run_event(callbacks=callbacks, event="on_stage_end")

    def infer(
            self, *,
            loaders: Dict[str, data.DataLoader],
            callbacks: Dict[str, Callback],
            epochs: int = 1, verbose: bool = False):
        """
        Main method for predicting with DL models.

        :param loaders: OrderedDict or torch DataLoaders to run on
        :param callbacks: OrderedDict of callback to use
        :param epochs: number of epochs to run
        :param verbose: verbose flag
        """
        return self.run(
            loaders=loaders, callbacks=callbacks,
            epochs=epochs, mode="infer", verbose=verbose)

    def batch_handler(self, *, dct, model, state) -> Dict:
        """
        Batch handler wrapper with main statistics and device management.

        :param dct: key-value storage with input tensors
        :param model: model to predict with
        :param state: runner state
        :return: key-value storage with model predictions
        """
        dct = {
            key: value.to(state.device)
            for key, value in dct.items()}
        state.input = dct
        state.bs = len(dct[list(dct.keys())[0]])  # @TODO: fixme
        output = self._batch_handler(dct=dct, model=model)
        return output

    @staticmethod
    def _batch_handler(*, dct, model) -> Dict:
        """
        Batch handler with model forward.

        :param dct: key-value storage with input tensors
        :param model: model to predict with
        :return: key-value storage with model predictions
        """
        raise NotImplementedError

    @staticmethod
    def prepare_callbacks(
            *,
            loggers : Dict[str, SummaryWriter],
            callbacks_params : Dict[str, Dict],
            args : Namespace,
            mode : str,
            stage : str = None) -> Dict[str, Callback]:
        """
        Runner callbacks method to handle different runs logic.

        :param loggers: SummaryWriter logger for LoggerCallback
        :param callbacks_params: parameters for callbacks creation
        :param args: console args
        :param mode: train/infer
        :param stage: training stage name
        :return: OrderedDict with callbacks
        """
        raise NotImplementedError
