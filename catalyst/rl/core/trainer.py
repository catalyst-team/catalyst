from typing import Dict, List, Optional  # isort:skip

import gc
import logging
from collections import OrderedDict
import os
from pathlib import Path
import shutil
import time

import numpy as np

import torch
from torch.utils.data import DataLoader

from catalyst import utils
from catalyst.dl import DLRunner, Experiment, DLState, \
    Callback, LoggerCallback
from .algorithm import AlgorithmSpec
from .db import DBSpec
from .environment import EnvironmentSpec

logger = logging.getLogger(__name__)


class TrainerSpec:
    def __init__(
        self,
        algorithm: AlgorithmSpec,
        environment_spec: EnvironmentSpec,
        logdir: str,
        num_workers: int = 1,
        batch_size: int = 64,
        min_num_transitions: int = int(1e4),
        online_update_period: int = 1,
        weights_sync_period: int = 1,
        seed: int = 42,
        epoch_limit: int = None,
        db_server: DBSpec = None,
        **kwargs,
    ):
        # environment & algorithm
        self.environment_spec = environment_spec
        self.algorithm = algorithm

        # logging
        self.logdir = logdir
        self._seeder = utils.Seeder(init_seed=seed)

        # updates & counters
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.epoch = 0
        self.update_step = 0
        self.num_updates = 0
        self._num_trajectories = 0
        self._num_transitions = 0

        # updates configuration
        # (actor_period, critic_period)
        self.actor_grad_period, self.critic_grad_period = \
            utils.make_tuple(online_update_period)

        # synchronization configuration
        self.db_server = db_server
        self.min_num_transitions = min_num_transitions
        self.weights_sync_period = weights_sync_period

        self.replay_buffer = None
        self.replay_sampler = None
        self.loader = None
        self._epoch_limit = epoch_limit or np.iinfo(np.int32).max

        #  special
        self._prepare_seed()

    def _prepare_for_stage(self, stage: str):
        utils.set_global_seed(self.experiment.initial_seed)
        migrating_params = {}
        if self.state is not None:
            migrating_params.update(
                {
                    "step": self.state.step,
                    "epoch": self.state.epoch
                }
            )

        self.model, criterion, optimizer, scheduler, self.device = \
            self._get_experiment_components(stage)

        self.state = DLState(
            stage=stage,
            model=self.model,
            device=self.device,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            **self.experiment.get_state_params(stage),
            **migrating_params
        )
        utils.set_global_seed(self.experiment.initial_seed)

    def _run_event(self, event: str, moment: Optional[str]):
        fn_name = f"on_{event}"
        if moment is not None:
            fn_name = f"{fn_name}_{moment}"

        # before callbacks
        if self.state is not None:
            getattr(self.state, f"{fn_name}_pre")()

        if self.loggers is not None and moment == "start":
            for logger in self.loggers.values():
                getattr(logger, fn_name)(self.state)

        # running callbacks
        if self.callbacks is not None:
            for callback in self.callbacks.values():
                getattr(callback, fn_name)(self.state)

        # after callbacks
        if self.loggers is not None and \
                (moment == "end" or moment is None):  # for on_exception case
            for logger in self.loggers.values():
                getattr(logger, fn_name)(self.state)

        if self.state is not None:
            getattr(self.state, f"{fn_name}_post")()

    def _update_target_weights(self, update_step) -> Dict:
        pass

    def _run_loader(self, loader: DataLoader) -> Dict:
        start_time = time.time()

        # @TODO: add average meters
        for batch in loader:
            metrics: Dict = self.algorithm.train(
                batch,
                actor_update=(self.update_step % self.actor_grad_period == 0),
                critic_update=(
                    self.update_step % self.critic_grad_period == 0
                )
            ) or {}
            self.update_step += 1

            metrics_ = self._update_target_weights(self.update_step) or {}
            metrics.update(**metrics_)

            metrics = dict(
                (key, value)
                for key, value in metrics.items()
                if isinstance(value, (float, int))
            )

        return output

    def _run_epoch(self) -> Dict:
        raise NotImplementedError()

    def _run_epoch_loop(self):
        self._prepare_seed()
        metrics: Dict = self._run_epoch()
        self.epoch += 1

    def _run_train_stage(self):
        self._run_event("stage", moment="start")
        while self.epoch < self._epoch_limit:
            self._run_epoch()
        self._run_event("stage", moment="end")

    def run_experiment(self, experiment: Experiment):
        self.experiment = experiment

        stage = "train"
        self._prepare_for_stage(stage)

        callbacks = self.experiment.get_callbacks(stage)
        loggers = utils.process_callbacks(
            OrderedDict([
                (k, v) for k, v in callbacks.items()
                if isinstance(v, LoggerCallback)
            ])
        )
        callbacks = utils.process_callbacks(
            OrderedDict([
                (k, v) for k, v in callbacks.items()
                if not isinstance(v, LoggerCallback)
            ])
        )
        self.state.loggers = loggers
        self.loggers = loggers
        self.callbacks = callbacks

        self._run_train_stage()
