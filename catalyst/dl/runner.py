import sys
import tqdm
from collections import OrderedDict
from argparse import Namespace
from typing import Dict
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

from catalyst.contrib.registry import Registry
from catalyst.dl.utils import UtilsFactory
from catalyst.utils.misc import merge_dicts
from catalyst.dl.callbacks import Callback
from catalyst.dl.datasource import AbstractDataSource
from catalyst.dl.state import RunnerState
from catalyst.dl.fp16 import Fp16Wrap

STAGE_KEYWORDS = [
    "criterion_params", "optimizer_params", "scheduler_params", "stage_params",
    "state_params", "data_params", "callbacks_params"
]


class BaseModelRunner:
    """
    Abstract model run handler.
    Based on model, it's criterion, optimizer and scheduler stuff.
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module = None,
        optimizer: optim.Optimizer = None,
        scheduler: optim.lr_scheduler._LRScheduler = None
    ):
        """

        :param model: nn.Module instance, your model
        :param criterion: torch criterion
        :param optimizer:  torch optimizer
        :param scheduler: torch scheduler
        """
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
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

    def _init_state(
        self, *, mode: str, stage: str = None, **kwargs
    ) -> RunnerState:
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
            criterion=self.criterion,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            **kwargs,
            **additional_kwargs
        )

    def run_event(self, *, callbacks: Dict[str, Callback], event: str):
        """
        Innert method to run special event for all available callbacks.

        :param callbacks:
        :param event:
        """
        getattr(self.state, f"{event}_pre")(state=self.state)
        for callback in callbacks.values():
            getattr(callback, event)(state=self.state)
        getattr(self.state, f"{event}_post")(state=self.state)

    def run(
        self,
        *,
        loaders: Dict[str, data.DataLoader],
        callbacks: Dict[str, Callback],
        state_params: Dict = None,
        epochs: int = 1,
        start_epoch: int = 0,
        mode: str = "train",
        verbose: bool = False,
        logdir: str = None
    ):
        """
        Main method for running train/valid/infer/debug pipeline over model.

        :param loaders: OrderedDict of torch DataLoaders to run on
        :param callbacks: OrderedDict of callback to use
        :param state_params: params for state initialization
        :param epochs: number of epochs to run
        :param start_epoch:
        :param mode: mode - train/infer/debug
        :param verbose: boolean flag for tqdm progress bar
        """
        assert isinstance(loaders, OrderedDict)
        assert isinstance(callbacks, OrderedDict)

        state_params = state_params or {}
        state = self._init_state(
            mode=mode, stage=self.stage, logdir=logdir, **state_params
        )
        state.mode = mode
        self.state = state

        self.run_event(callbacks=callbacks, event=f"on_{mode}_start")
        for epoch in range(start_epoch, start_epoch + epochs):
            state.epoch = epoch
            self.run_event(callbacks=callbacks, event="on_epoch_start")

            for loader_mode, loader in loaders.items():
                state.loader_mode = loader_mode
                state.is_train = loader_mode.startswith("train")
                state.batch_size = loader.batch_size
                state.loader_len = len(loader)
                state.step = (
                    state.step or state.epoch * len(loader) * state.batch_size
                )
                self.model.train(state.is_train)

                self.run_event(callbacks=callbacks, event="on_loader_start")
                loader = tqdm.tqdm(
                    loader,
                    desc=f"{epoch} * Epoch ({loader_mode})",
                    total=len(loader),
                    leave=True,
                    file=sys.stdout,
                    ncols=0,
                ) if verbose else loader

                for i, dct in enumerate(loader):
                    dct = self.batch2device(dct=dct, state=state)
                    state.input = dct

                    self.run_event(callbacks=callbacks, event="on_batch_start")
                    with torch.set_grad_enabled(state.is_train):
                        state.output = self.batch_handler(
                            dct=state.input, model=self.model, state=state
                        )
                    self.run_event(callbacks=callbacks, event="on_batch_end")

                    if verbose:
                        loader.set_postfix(
                            **{
                                k: "{:.5f}".format(v)
                                for k, v in
                                sorted(state.batch_metrics.items())
                                if not k.startswith("base")
                            }
                        )
                        loader.update()

                self.run_event(callbacks=callbacks, event="on_loader_end")

            self.run_event(callbacks=callbacks, event="on_epoch_end")

        self.run_event(callbacks=callbacks, event=f"on_{mode}_end")

    def train(
        self,
        *,
        loaders: Dict[str, data.DataLoader],
        callbacks: Dict[str, Callback],
        state_params: Dict = None,
        epochs: int = 1,
        start_epoch: int = 0,
        verbose: bool = False,
        logdir: str = None
    ):
        """
        One stage training method.

        :param loaders: OrderedDict of torch DataLoaders to run on
        :param callbacks: OrderedDict of callback to use
        :param state_params: params for state initialization
        :param epochs: number of epochs to run
        :param start_epoch:
        :param verbose: verbose flag
        :param logdir: logdir for tensorboard logs
        """
        self.run(
            loaders=loaders,
            callbacks=callbacks,
            state_params=state_params,
            epochs=epochs,
            start_epoch=start_epoch,
            mode="train",
            verbose=verbose,
            logdir=logdir,
        )

    @staticmethod
    def prepare_stage_args(*, args, stage_config):
        for key, value in stage_config.get("args", {}).items():
            setattr(args, key, value)
        return args

    @staticmethod
    def prepare_stage_model(*, model, stage, **kwargs):
        assert len(kwargs) == 0
        pass

    @staticmethod
    def prepare_model_stuff(
        *,
        model,
        criterion_params=None,
        optimizer_params=None,
        scheduler_params=None
    ):
        fp16 = isinstance(model, Fp16Wrap)

        criterion_params = criterion_params or {}
        criterion = Registry.get_criterion(**criterion_params)

        optimizer_params = optimizer_params or {}
        optimizer = Registry.get_optimizer(
            model, **optimizer_params, fp16=fp16
        )

        scheduler_params = scheduler_params or {}
        scheduler = Registry.get_scheduler(optimizer, **scheduler_params)

        return criterion, optimizer, scheduler

    def train_stages(
        self,
        *,
        datasource: AbstractDataSource,
        args: Namespace,
        stages_config: Dict[str, Dict] = None,
        verbose: bool = False
    ):
        """
        Main method for training DL models.

        :param datasource: AbstractDataSource instance
        :param args: console args
        :param stages_config: config
        :param verbose: verbose flag
        """

        stages_params = {}
        for key in STAGE_KEYWORDS:
            stages_params[key] = stages_config.pop(key, {})
        loaders = None

        for stage, config in stages_config.items():
            self.stage = stage

            args = self.prepare_stage_args(args=args, stage_config=config)

            for key in STAGE_KEYWORDS:
                config[key] = merge_dicts(
                    stages_params[key], config.get(key, {})
                )

            reload_loaders = config["data_params"].pop("reload_loaders", True)

            if loaders is None or reload_loaders:
                loaders = datasource.prepare_loaders(
                    mode="train",
                    stage=stage,
                    n_workers=args.workers,
                    batch_size=args.batch_size,
                    **config.pop("data_params")
                )

            callbacks = self.prepare_callbacks(
                mode="train",
                stage=stage,
                resume=args.resume,
                **config.pop("callbacks_params")
            )

            self.prepare_stage_model(
                model=self.model, stage=stage, **config.pop("stage_params")
            )
            self.criterion, self.optimizer, self.scheduler = \
                self.prepare_model_stuff(
                    model=self.model,
                    criterion_params=config.pop("criterion_params"),
                    optimizer_params=config.pop("optimizer_params"),
                    scheduler_params=config.pop("scheduler_params"))

            start_epoch = 0 if self.state is None else self.state.epoch + 1
            self.train(
                loaders=loaders,
                callbacks=callbacks,
                state_params=config.pop("state_params"),
                epochs=args.epochs,
                start_epoch=start_epoch,
                verbose=verbose,
                logdir=args.logdir
            )

    def infer(
        self,
        *,
        loaders: Dict[str, data.DataLoader],
        callbacks: Dict[str, Callback],
        epochs: int = 1,
        verbose: bool = False
    ):
        """
        Main method for predicting with DL models.

        :param loaders: OrderedDict of torch DataLoaders to run on
        :param callbacks: OrderedDict of callback to use
        :param epochs: number of epochs to run
        :param verbose: verbose flag
        """
        return self.run(
            loaders=loaders,
            callbacks=callbacks,
            epochs=epochs,
            mode="infer",
            verbose=verbose
        )

    def batch_handler(
        self, *, dct: Dict, model: nn.Module, state: RunnerState = None
    ) -> Dict:
        """
        Batch handler wrapper with main statistics and device management.

        :param dct: key-value storage with input tensors
        :param model: model to predict with
        :param state: runner state
        :return: key-value storage with model predictions
        """
        dct = self.batch2device(dct=dct, state=state)
        output = self._batch_handler(dct=dct, model=model)
        return output

    def batch2device(self, *, dct: Dict, state: RunnerState = None):
        if state is not None:
            dct = {
                key: value.to(self.device)
                if state.key2device[key] and torch.is_tensor(value) else value
                for key, value in dct.items()
            }
        else:
            dct = {key: value.to(self.device) for key, value in dct.items()}
        return dct

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
    def prepare_callbacks(
        *,
        mode: str,
        stage: str = None,
        resume: str = None,
        out_prefix: str = None,
        **kwargs
    ) -> Dict[str, Callback]:
        """
        Runner callbacks method to handle different runs logic.

        :param args: console args
        :param mode: train/infer
        :param stage: training stage name
        :param resume: path to checkpoint (used for checkpoint callback)
        :param **kwargs: callbacks params
        :return: OrderedDict with callbacks
        """
        callbacks = OrderedDict()

        for key, value in kwargs.items():
            callback = Registry.get_callback(**value)
            callbacks[key] = callback

        for key, value in callbacks.items():
            # @TODO: remove hack
            if resume is not None and hasattr(value, "resume"):
                value.resume = resume
            if out_prefix is not None and hasattr(value, "out_prefix"):
                value.out_prefix = out_prefix

        return callbacks


class SupervisedModelRunner(BaseModelRunner):
    def batch2device(self, *, dct: Dict, state: RunnerState = None):
        if isinstance(dct, (tuple, list)):
            assert len(dct) == 2
            dct = {"features": dct[0], "targets": dct[1]}
        dct = super().batch2device(dct=dct, state=state)
        return dct

    def batch_handler(
        self, *, dct: Dict, model: nn.Module, state: RunnerState = None
    ) -> Dict:
        """
        Batch handler wrapper with main statistics and device management.

        :param dct: key-value storage with input tensors
        :param model: model to predict with
        :param state: runner state
        :return: key-value storage with model predictions
        """
        dct = self.batch2device(dct=dct, state=state)
        logits = model(dct["features"])
        output = {"logits": logits}

        return output
