from typing import List, Dict

import os
import gc
import time
import shutil
from pathlib import Path
import numpy as np
import logging

from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from catalyst.utils.seed import set_global_seed, Seeder
from catalyst import utils
from .db import DBSpec
from .environment import EnvironmentSpec
from .algorithm import AlgorithmSpec

logger = logging.getLogger(__name__)

if os.environ.get("USE_WANDB", "1") == "1":
    try:
        import wandb
        WANDB_ENABLED = True
    except ImportError:
        logger.warning(
            "wandb not available, to install wandb, run `pip install wandb`."
        )
        WANDB_ENABLED = False
else:
    WANDB_ENABLED = False


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
        save_period: int = 10,
        gc_period: int = 10,
        seed: int = 42,
        epoch_limit: int = None,
        monitoring_params: Dict = None,
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
        self.save_period = save_period
        self.weights_sync_period = weights_sync_period

        self._gc_period = gc_period

        self.replay_buffer = None
        self.replay_sampler = None
        self.loader = None
        self._epoch_limit = epoch_limit

        #  special
        self.monitoring_params = monitoring_params
        self._prepare_seed()
        self._init(**kwargs)

    def _init(self, **kwargs):
        global WANDB_ENABLED
        assert len(kwargs) == 0
        if WANDB_ENABLED:
            if self.monitoring_params is not None:
                self.checkpoints_glob: List[str] = \
                    self.monitoring_params.pop(
                        "checkpoints_glob", ["best.pth", "last.pth"])

                wandb.init(**self.monitoring_params)

                logdir_src = Path(self.logdir)
                logdir_dst = Path(wandb.run.dir)

                configs_src = logdir_src.joinpath("configs")
                os.makedirs(f"{logdir_dst}/{configs_src.name}", exist_ok=True)
                shutil.rmtree(f"{logdir_dst}/{configs_src.name}")
                shutil.copytree(
                    f"{str(configs_src.absolute())}",
                    f"{logdir_dst}/{configs_src.name}")

                code_src = logdir_src.joinpath("code")
                if code_src.exists():
                    os.makedirs(f"{logdir_dst}/{code_src.name}", exist_ok=True)
                    shutil.rmtree(f"{logdir_dst}/{code_src.name}")
                    shutil.copytree(
                        f"{str(code_src.absolute())}",
                        f"{logdir_dst}/{code_src.name}")
            else:
                WANDB_ENABLED = False
        self.wandb_mode = "trainer"

    def _prepare_logger(self, logdir):
        timestamp = utils.get_utcnow_time()
        logpath = f"{logdir}/trainer.{timestamp}"
        os.makedirs(logpath, exist_ok=True)
        self.logger = SummaryWriter(logpath)

    def _prepare_seed(self):
        seed = self._seeder()[0]
        set_global_seed(seed)

    def _log_to_console(
        self, fps: float, updates_per_sample: float, num_trajectories: int,
        num_transitions: int, buffer_size: int, **kwargs
    ):
        prefix = f"--- Epoch {self.epoch:09d}/{self._epoch_limit:09d}" \
            if self._epoch_limit is not None \
            else f"--- Epoch {self.epoch:09d}"
        metrics = [
            prefix,
            f"fps: {fps:7.1f}",
            f"updates per sample: {updates_per_sample:7.1f}",
            f"trajectories: {num_trajectories:09d}",
            f"transitions: {num_transitions:09d}",
            f"buffer size: {buffer_size:09d}",
        ]
        metrics = " | ".join(metrics)
        print(metrics)

    def _log_to_tensorboard(
        self, fps: float, updates_per_sample: float, num_trajectories: int,
        num_transitions: int, buffer_size: int, **kwargs
    ):
        self.logger.add_scalar("fps", fps, self.epoch)
        self.logger.add_scalar(
            "updates_per_sample", updates_per_sample, self.epoch
        )
        self.logger.add_scalar(
            "num_trajectories", num_trajectories, self.epoch
        )
        self.logger.add_scalar("num_transitions", num_transitions, self.epoch)
        self.logger.add_scalar("buffer_size", buffer_size, self.epoch)
        self.logger.flush()

    @staticmethod
    def _log_wandb_metrics(
        metrics: Dict,
        step: int,
        mode: str,
        suffix: str = ""
    ):
        metrics = {
            f"{mode}/{key}{suffix}": value
            for key, value in metrics.items()
        }
        step = None  # @TODO: fix, wandb issue
        wandb.log(metrics, step=step)

    def _log_to_wandb(self, *, step, suffix="", **metrics):
        if WANDB_ENABLED:
            self._log_wandb_metrics(
                metrics, step=step, mode=self.wandb_mode, suffix=suffix)

    def _save_wandb(self):
        if WANDB_ENABLED:
            logdir_src = Path(self.logdir)
            logdir_dst = Path(wandb.run.dir)

            events_src = list(logdir_src.glob("events.out.tfevents*"))
            if len(events_src) > 0:
                events_src = events_src[0]
                os.makedirs(f"{logdir_dst}/{logdir_src.name}", exist_ok=True)
                shutil.copy2(
                    f"{str(events_src.absolute())}",
                    f"{logdir_dst}/{logdir_src.name}/{events_src.name}")

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
            checkpoint = self.algorithm.pack_checkpoint(with_optimizer=False)
            for key in checkpoint:
                checkpoint[key] = {
                    k: v.detach().cpu().numpy()
                    for k, v in checkpoint[key].items()
                }

            self.db_server.put_checkpoint(
                checkpoint=checkpoint, epoch=self.epoch
            )

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

            for key, value in metrics.items():
                self.logger.add_scalar(key, value, self.update_step)
            self._log_to_wandb(
                step=self.update_step, suffix="_batch", **metrics)

        elapsed_time = time.time() - start_time
        elapsed_num_updates = len(loader) * loader.batch_size
        self.num_updates += elapsed_num_updates
        fps = elapsed_num_updates / elapsed_time

        output = {"elapsed_time": elapsed_time, "fps": fps}

        return output

    def _run_epoch(self) -> Dict:
        raise NotImplementedError()

    def _run_epoch_loop(self):
        self._prepare_seed()
        metrics: Dict = self._run_epoch()
        self.epoch += 1
        self._log_to_console(**metrics)
        self._log_to_tensorboard(**metrics)
        self._log_to_wandb(step=self.epoch, suffix="_epoch", **metrics)
        self._save_checkpoint()
        self._save_wandb()
        self._update_sampler_weights()
        if self.epoch % self._gc_period == 0:
            gc.collect()

    def _run_train_loop(self):
        self.db_server.push_message(self.db_server.Message.ENABLE_TRAINING)
        epoch_limit = self._epoch_limit or np.iinfo(np.int32).max
        while self.epoch < epoch_limit:
            try:
                self._run_epoch_loop()
            except Exception as ex:
                self.db_server.push_message(
                    self.db_server.Message.DISABLE_TRAINING)
                raise ex
        self.db_server.push_message(self.db_server.Message.DISABLE_TRAINING)

    def _start_train_loop(self):
        self._run_train_loop()

    def run(self):
        self._update_sampler_weights()
        self._start_train_loop()
        self.logger.close()
