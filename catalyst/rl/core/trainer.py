from typing import Dict

import os
import gc
import time
from datetime import datetime

from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from catalyst.utils.seed import set_global_seed, Seeder
from catalyst import utils
from .db import DBSpec
from .environment import EnvironmentSpec
from .algorithm import AlgorithmSpec


class TrainerSpec:
    def __init__(
        self,
        algorithm: AlgorithmSpec,
        env_spec: EnvironmentSpec,
        db_server: DBSpec,
        logdir: str,
        num_workers: int = 1,
        batch_size: int = 64,
        min_num_transitions: int = int(1e4),
        online_update_period: int = 1,
        weights_sync_period: int = 1,
        weights_sync_mode: str = None,
        save_period: int = 10,
        gc_period: int = 10,
        seed: int = 42,
        **kwargs,
    ):
        # algorithm & environment
        self.algorithm = algorithm
        self.env_spec = env_spec

        # logging
        self.logdir = logdir
        self._prepare_logger(logdir)
        self._seeder = Seeder(init_seed=seed)

        # updates & counters
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.epoch = 0
        self.step = 0
        self._num_trajectories = 0
        self._num_transitions = 0

        # updates configuration
        # (actor_period, critic_period)
        self.actor_grad_period, self.critic_grad_period = \
            utils.make_tuple(online_update_period)

        # synchronization configuration
        self.db_server = db_server
        self.min_num_transitions = min_num_transitions
        self.save_period = save_period
        self.weights_sync_period = weights_sync_period

        self._weights_sync_mode = weights_sync_mode
        self._gc_period = gc_period

        self.replay_buffer = None
        self.replay_sampler = None
        self.loader = None

        #  special
        self._init(**kwargs)

    def _init(self, **kwargs):
        assert len(kwargs) == 1

    def _prepare_logger(self, logdir):
        if logdir is not None:
            timestamp = datetime.utcnow().strftime("%y%m%d.%H%M%S")
            logpath = f"{logdir}/trainer.{timestamp}"
            os.makedirs(logpath, exist_ok=True)
            self.logger = SummaryWriter(logpath)
        else:
            self.logger = None

    def _prepare_seed(self):
        seed = self._seeder()[0]
        set_global_seed(seed)

    def _log_to_console(
        self,
        fps: float,
        num_trajectories: int,
        num_transitions: int,
        buffer_size: int,
        **kwargs
    ):
        print(
            "--- "
            f"fps: {fps:5.1f}\t"
            f"trajectories: {num_trajectories:09d}\t"
            f"transitions: {num_transitions:09d}\t"
            f"buffer size: {buffer_size:09d}"
        )

    def _log_to_tensorboard(
        self,
        fps: float,
        num_trajectories: int,
        num_transitions: int,
        buffer_size: int,
        **kwargs
    ):
        self.logger.add_scalar("fps", fps, self.epoch)
        self.logger.add_scalar(
            "num_trajectories", num_trajectories, self.epoch)
        self.logger.add_scalar(
            "num_transitions", num_transitions, self.epoch)
        self.logger.add_scalar("buffer_size", buffer_size, self.epoch)

    def _save_checkpoint(self):
        if self.epoch % self.save_period == 0:
            checkpoint = self.algorithm.pack_checkpoint()
            checkpoint["epoch"] = self.epoch
            filename = utils.save_checkpoint(
                logdir=self.logdir,
                checkpoint=checkpoint,
                suffix=str(self.epoch)
            )
            print(f"Checkpoint saved to: {filename}")

    def _update_sampler_weights(self):
        if self.epoch % self.weights_sync_period == 0:
            state_dict = self.algorithm.__dict__[
                self._weights_sync_mode].state_dict()
            state_dict = {
                k: v.detach().cpu().numpy()
                for k, v in state_dict.items()
            }
            self.db_server.dump_weights(
                weights=state_dict,
                prefix=self._weights_sync_mode,
                epoch=self.epoch
            )

    def _update_target_weights(self, step):
        pass

    def _run_loader(self, loader: DataLoader) -> Dict:
        start_time = time.time()
        start_step = self.step

        # @TODO: add average meters
        for batch in loader:
            metrics: Dict = self.algorithm.train(
                batch,
                actor_update=(self.step % self.actor_grad_period == 0),
                critic_update=(self.step % self.critic_grad_period == 0)
            ) or {}
            self.step += 1

            metrics_: Dict = self._update_target_weights(self.step) or {}
            metrics.update(**metrics_)

            for key, value in metrics.items():
                if isinstance(value, (float, int)):
                    self.logger.add_scalar(key, value, self.step)

        elapsed_time = time.time() - start_time
        elapsed_step = self.step - start_step
        fps = loader.batch_size * elapsed_step / elapsed_time

        output = {
            "elapsed_time": elapsed_time,
            "fps": fps
        }

        return output

    def _run_epoch(self) -> Dict:
        raise NotImplementedError()

    def _run_epoch_loop(self):
        self._prepare_seed()
        metrics: Dict = self._run_epoch()
        self.epoch += 1
        self._save_checkpoint()
        self._log_to_console(**metrics)
        self._log_to_tensorboard(**metrics)
        self._update_sampler_weights()
        if self.epoch % self._gc_period == 0:
            gc.collect()

    def _run_train_loop(self):
        while True:
            self._run_epoch_loop()

    def _start_train_loop(self):
        self._run_train_loop()

    def run(self):
        self._update_sampler_weights()
        self._start_train_loop()
