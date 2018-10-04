import tqdm
from pprint import pprint
from collections import OrderedDict
from argparse import Namespace
from typing import Dict
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

from prometheus.utils.factory import UtilsFactory
from prometheus.utils.misc import merge_dicts
from prometheus.dl.callbacks import Callback
from prometheus.dl.datasource import AbstractDataSource
from prometheus.dl.state import RunnerState


class AbstractModelRunner:
    """
    Abstract model run handler.
    Based on model, it's criterion, optimizer and scheduler stuff.
    """

    def __init__(self,
                 model: nn.Module,
                 criterion: Dict[str, nn.Module] = None,
                 optimizer: Dict[str, optim.Optimizer] = None,
                 scheduler: Dict[str, optim.lr_scheduler._LRScheduler] = None,
                 debug: bool = True):
        """

        :param model: nn.Module instance, your model
        :param criterion: OrderedDict with torch criterions for model training
        :param optimizer: OrderedDict with torch optimizers for model training
        :param scheduler: OrderedDict with torch schedulers for optimizers lrs
        :param debug: boolean flag for all info printing
        """
        self.model = model
        self.criterion = criterion or {}
        self.optimizer = optimizer or {}
        self.scheduler = scheduler or {}

        stuff_handler = lambda x: {"main": x} if not isinstance(x, dict) else x

        self.criterion = stuff_handler(self.criterion)
        self.optimizer = stuff_handler(self.optimizer)
        self.scheduler = stuff_handler(self.scheduler)

        if debug:
            pprint(model)
            pprint(criterion)
            pprint(optimizer)
            pprint(scheduler)

        self.device = None
        self.state = None
        self.stage = None
        self._init()

    def _init(self):
        """
        Inner method for children's classes for model specific initialization.
        As baseline, checks device support and puts model on it.
        :return:
        """
        self.model, self.device = UtilsFactory.prepare_model(self.model)

    def _init_state(self, *, mode: str, stage: str = None,
                    **kwargs) -> RunnerState:
        """
        Inner method for children's classes for state specific initialization.
        :return: RunnerState with all necessary parameters.
        """
        additional_kwargs = {}

        # transfer previous counters from old state
        if self.state is not None:
            additional_kwargs = {
                "step": self.state.step,
                "epoch": self.state.epoch + 1,
                "best_metrics": self.state.best_metrics
            }
        return RunnerState(
            device=self.device,
            model=self.model,
            stage=self.stage,
            _criterion=self.criterion,
            _optimizer=self.optimizer,
            _scheduler=self.scheduler,
            **kwargs,
            **additional_kwargs)

    def run_stage_init(self, callbacks: Dict[str, Callback]):
        for callback in callbacks.values():
            callback.on_stage_init(model=self.model, stage=self.stage)

    def run_event(self, *, callbacks: Dict[str, Callback], event: str):
        """
        Innert method to run special event for all available callbacks.

        :param callbacks:
        :param event:
        """
        for callback in callbacks.values():
            getattr(callback, event)(state=self.state)

    def run(self,
            *,
            loaders: Dict[str, data.DataLoader],
            callbacks: Dict[str, Callback],
            epochs: int = 1,
            start_epoch: int = 0,
            mode: str = "train",
            verbose: bool = False):
        """
        Main method for running train/valid/infer/debug pipeline over model.

        :param loaders: OrderedDict or torch DataLoaders to run on
        :param callbacks: OrderedDict of callback to use
        :param epochs: number of epochs to run
        :param start_epoch:
        :param mode: mode - train/infer/debug
        :param verbose: boolean flag for tqdm progress bar
        """
        assert isinstance(loaders, OrderedDict)
        assert isinstance(callbacks, OrderedDict)

        self.state = self._init_state(mode=mode, stage=self.stage)

        self.run_event(callbacks=callbacks, event=f"on_{mode}_start")

        for epoch in range(start_epoch, start_epoch + epochs):
            self.state.epoch = epoch

            self.run_event(callbacks=callbacks, event="on_epoch_start")

            for loader_mode, loader in loaders.items():
                self.state.batch_size = loader.batch_size
                self.state.loader_mode = loader_mode
                self.state.loader = loader

                self.state.is_train = loader_mode.startswith("train")
                self.model.train(self.state.is_train)

                self.run_event(callbacks=callbacks, event="on_loader_start")

                self.state.loader = tqdm.tqdm(
                    self.state.loader,
                    total=len(loader),
                    desc=f"Epoch {epoch}  training",
                    ncols=0) if verbose else self.state.loader

                for i, dct in enumerate(self.state.loader):
                    self.state.input = dct

                    self.run_event(callbacks=callbacks, event="on_batch_start")

                    with torch.set_grad_enabled(self.state.is_train):
                        self.state.output = self.batch_handler(
                            dct=dct, model=self.model, state=self.state)

                    self.run_event(callbacks=callbacks, event="on_batch_end")

                self.run_event(callbacks=callbacks, event="on_loader_end")

            self.run_event(callbacks=callbacks, event="on_epoch_end")

        self.run_event(callbacks=callbacks, event=f"on_{mode}_end")

    def train_stage(self,
                    *,
                    loaders: Dict[str, data.DataLoader],
                    callbacks: Dict[str, Callback],
                    epochs: int = 1,
                    start_epoch: int = 0,
                    verbose: bool = False,
                    logdir: str = None):
        """
        One stage training method.

        :param loaders: OrderedDict or torch DataLoaders to run on
        :param callbacks: OrderedDict of callback to use
        :param epochs: number of epochs to run
        :param start_epoch:
        :param verbose: verbose flag
        :param logdir: logdir for tensorboard logs
        """
        # @TODO: better solution
        if logdir is not None:
            loggers = UtilsFactory.create_loggers(logdir, loaders)
            for key, value in callbacks.items():
                if hasattr(value, "loggers"):
                    value.loggers = loggers
                if hasattr(value, "logdir"):
                    value.logdir = logdir
        self.run(
            loaders=loaders,
            callbacks=callbacks,
            epochs=epochs,
            start_epoch=start_epoch,
            mode="train",
            verbose=verbose)

    def train(self,
              *,
              datasource: AbstractDataSource,
              args: Namespace,
              stages_config: Dict[str, Dict] = None,
              verbose: bool = False):
        """
        Main method for training DL models.

        :param datasource: AbstractDataSource instance
        :param args: console args
        :param stages_config: config
        :param verbose: verbose flag
        """

        stages_data_params = stages_config.pop("data_params", {})
        stages_callbacks_params = stages_config.pop("callbacks_params", {})
        stages_criterion_params = stages_config.pop("criterion_params", {})
        stages_optimizer_params = stages_config.pop("optimizer_params", {})
        loaders = None

        for stage, config in stages_config.items():
            self.stage = stage

            args = UtilsFactory.prepare_stage_args(
                args=args, stage_config=config)
            pprint(args)

            data_params = merge_dicts(stages_data_params,
                                      config.get("data_params", {}))
            reload_loaders = data_params.get("reload_loaders", True)

            if loaders is None or reload_loaders:
                loaders = datasource.prepare_loaders(
                    args, data_params, stage=stage)

            callbacks_params = merge_dicts(stages_callbacks_params,
                                           config.get("callbacks_params", {}))
            config["criterion_params"] = merge_dicts(
                stages_criterion_params, config.get("criterion_params", {}))
            config["optimizer_params"] = merge_dicts(
                stages_optimizer_params, config.get("optimizer_params", {}))

            callbacks = self.prepare_callbacks(
                callbacks_params=callbacks_params,
                args=args,
                mode="train",
                stage=stage)
            pprint(loaders)
            pprint(callbacks)

            self.run_stage_init(callbacks=callbacks)
            self.criterion, self.optimizer, self.scheduler = \
                UtilsFactory.prepare_stage_stuff(
                    model=self.model, stage_config=config)

            self.train_stage(
                loaders=loaders,
                callbacks=callbacks,
                epochs=args.epochs,
                start_epoch=0 if self.state is None else self.state.epoch + 1,
                verbose=verbose,
                logdir=args.logdir)

    def infer(self,
              *,
              loaders: Dict[str, data.DataLoader],
              callbacks: Dict[str, Callback],
              epochs: int = 1,
              verbose: bool = False):
        """
        Main method for predicting with DL models.

        :param loaders: OrderedDict or torch DataLoaders to run on
        :param callbacks: OrderedDict of callback to use
        :param epochs: number of epochs to run
        :param verbose: verbose flag
        """
        return self.run(
            loaders=loaders,
            callbacks=callbacks,
            epochs=epochs,
            mode="infer",
            verbose=verbose)

    def batch_handler(self,
                      *,
                      dct: Dict,
                      model: nn.Module,
                      state: RunnerState = None) -> Dict:
        """
        Batch handler wrapper with main statistics and device management.

        :param dct: key-value storage with input tensors
        :param model: model to predict with
        :param state: runner state
        :return: key-value storage with model predictions
        """
        dct = {key: value.to(self.device) for key, value in dct.items()}
        if state is not None:
            state.input = dct
        output = self._batch_handler(dct=dct, model=model)
        return output

    @staticmethod
    def _batch_handler(*, dct: Dict, model: nn.Module) -> Dict:
        """
        Batch handler with model forward.

        :param dct: key-value storage with model inputs
        :param model: model to predict with
        :return: key-value storage with model predictions
        """
        raise NotImplementedError

    @staticmethod
    def prepare_callbacks(*,
                          callbacks_params: Dict[str, Dict],
                          args: Namespace,
                          mode: str,
                          stage: str = None) -> Dict[str, Callback]:
        """
        Runner callbacks method to handle different runs logic.

        :param callbacks_params: parameters for callbacks creation
        :param args: console args
        :param mode: train/infer
        :param stage: training stage name
        :return: OrderedDict with callbacks
        """
        raise NotImplementedError


class ClassificationRunner(AbstractModelRunner):
    def batch_handler(self,
                      *,
                      dct: Dict,
                      model: nn.Module,
                      state: RunnerState = None) -> Dict:
        """
        Batch handler wrapper with main statistics and device management.

        :param dct: key-value storage with input tensors
        :param model: model to predict with
        :param state: runner state
        :return: key-value storage with model predictions
        """
        if isinstance(dct, (tuple, list)):
            assert len(dct) == 2
            dct = {"features": dct[0], "targets": dct[1]}
        dct = {key: value.to(state.device) for key, value in dct.items()}
        if state is not None:
            state.input = dct
        logits = model(dct["features"])
        output = {"logits": logits}
        return output
