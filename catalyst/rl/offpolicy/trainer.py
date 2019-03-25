import os
import time
from datetime import datetime
from tensorboardX import SummaryWriter
import multiprocessing as mp
import queue
import torch
from torch.utils.data import DataLoader

from catalyst.dl.utils import UtilsFactory
from catalyst.utils.serialization import serialize, deserialize
from catalyst.rl.offpolicy.utils import \
    ReplayBufferDataset, ReplayBufferSampler


def redis2queue_loop(redis, queue, max_size):
    pointer = 0
    redis_len = redis.llen("trajectories") - 1
    while True:
        try:
            need_more = pointer < redis_len and queue.qsize() < max_size
        except NotImplementedError:  # MacOS qsize issue (no sem_getvalue)
            need_more = pointer < redis_len

        if need_more:
            episode = deserialize(redis.lindex("trajectories", pointer))
            queue.put(episode, block=True, timeout=1.0)
            pointer += 1
        else:
            time.sleep(1.0)

        redis_len = redis.llen("trajectories") - 1


class Trainer:
    def __init__(
        self,
        algorithm,
        state_shape,
        action_shape,
        logdir,
        redis_server=None,
        redis_prefix=None,
        num_workers=1,
        replay_buffer_size=int(1e6),
        batch_size=64,
        start_learning=int(1e3),
        gamma=0.99,
        n_step=1,
        history_len=1,
        discrete_actions=False,
        epoch_len=int(1e2),
        save_period=10,
        target_update_period=1,
        online_update_period=1,
        weights_sync_period=1,
        max_redis_trials=1000,
        resume=None
    ):
        # algorithm
        self.algorithm = algorithm
        if resume is not None:
            self.algorithm.load_checkpoint(resume)

        # logging
        self.logdir = logdir
        current_date = datetime.now().strftime("%y-%m-%d-%H-%M-%S-%M-%f")
        logpath = f"{logdir}/trainer-{current_date}"
        os.makedirs(logpath, exist_ok=True)
        self.logger = SummaryWriter(logpath)

        # main attributes
        self.gamma = gamma
        self.n_step = n_step
        self.history_len = history_len
        self.discrete_actions = discrete_actions

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.epoch = 0
        self.epoch_len = epoch_len

        self.replay_buffer = ReplayBufferDataset(
            observation_shape=state_shape,
            action_shape=action_shape,
            max_size=replay_buffer_size,
            history_len=history_len,
            n_step=n_step,
            gamma=gamma,
            discrete_actions=discrete_actions
        )

        self.replay_sampler = ReplayBufferSampler(
            buffer=self.replay_buffer,
            epoch_len=epoch_len,
            batch_size=batch_size
        )

        self.loader = DataLoader(
            dataset=self.replay_buffer,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            sampler=self.replay_sampler
        )

        # updates configuration
        # (actor_period, critic_period)
        target_update_period = (
            target_update_period
            if isinstance(target_update_period, list)
            else (target_update_period, target_update_period)
        )
        self.actor_update_period, self.critic_update_period = \
            target_update_period

        self.actor_updates = 0
        self.critic_updates = 0

        # (actor_period, critic_period)
        online_update_period = (
            online_update_period
            if isinstance(online_update_period, list)
            else (online_update_period, online_update_period)
        )
        self.actor_grad_period, self.critic_grad_period = \
            online_update_period

        # synchronization configuration
        self.redis_server = redis_server
        self.redis_prefix = redis_prefix
        self.max_redis_trials = max_redis_trials
        self.start_learning = start_learning

        self.save_period = save_period
        self.weights_sync_period = weights_sync_period

        self.episodes_queue = mp.Queue()
        self._redis_loop_process = None

        self._sampler_weight_mode = \
            "critic" if self.discrete_actions else "actor"

    def save(self):
        if self.epoch % self.save_period == 0:
            checkpoint = self.algorithm.prepare_checkpoint()
            checkpoint["epoch"] = self.epoch
            filename = UtilsFactory.save_checkpoint(
                logdir=self.logdir,
                checkpoint=checkpoint,
                suffix=str(self.epoch)
            )
            print("Checkpoint saved to: %s" % filename)

    def get_processes(self):
        processes = []
        if self._redis_loop_process is not None:
            processes.append(self._redis_loop_process)

        return processes

    def _start_redis_loop(self):
        self._redis_loop_process = mp.Process(
            target=redis2queue_loop,
            kwargs={
                "redis": self.redis_server,
                "queue": self.episodes_queue,
                "max_size": int(self.max_redis_trials * 2)
            }
        )
        self._redis_loop_process.start()

    def _fetch_episodes(self):
        for i in range(self.max_redis_trials):
            try:
                episode = self.episodes_queue.get(block=True, timeout=1.0)
                self.replay_buffer.push_episode(episode)
            except queue.Empty:
                break
        stored = len(self.replay_buffer)
        print(f"transitions: {stored}")
        return i

    def _update_samplers_weights(self):
        if self.epoch % self.weights_sync_period == 0:
            mode = self._sampler_weight_mode
            state_dict = self.algorithm.__dict__[mode].state_dict()
            state_dict = {
                k: v.tolist()
                for k, v in state_dict.items()
            }
            self.redis_server.set(
                f"{self.redis_prefix}_{mode}_weights",
                serialize(state_dict)
            )

    def _update_target_weights(self, step_index):
        if not self.discrete_actions:
            if step_index % self.actor_update_period == 0:
                self.algorithm.target_actor_update()
                self.actor_updates += 1

                self.logger.add_scalar(
                    "actor updates", self.actor_updates, step_index
                )

        if step_index % self.critic_update_period == 0:
            self.algorithm.target_critic_update()
            self.critic_updates += 1

            self.logger.add_scalar(
                "critic updates", self.critic_updates, step_index
            )

    def _train(self):
        start_time = time.time()
        while True:
            fetch_i = self._fetch_episodes()
            elapsed_time = time.time() - start_time
            self.logger.add_scalar("fetch time", elapsed_time, self.epoch)
            self.logger.add_scalar("fetch index", fetch_i, self.epoch)
            start_time = time.time()

            for i, batch in enumerate(self.loader):
                step_index = self.epoch * self.epoch_len + i + 1
                metrics = self.algorithm.train(
                    batch,
                    actor_update=(step_index % self.actor_grad_period == 0),
                    critic_update=(step_index % self.critic_grad_period == 0)
                )

                for key, value in metrics.items():
                    if isinstance(value, float):
                        self.logger.add_scalar(key, value, step_index)

                self._update_target_weights(step_index)

            elapsed_time = time.time() - start_time
            self.logger.add_scalar("batch size", self.batch_size, self.epoch)
            self.logger.add_scalar(
                "buffer size", len(self.replay_buffer), self.epoch)
            self.logger.add_scalar(
                "batches per second", i / elapsed_time, self.epoch
            )
            self.logger.add_scalar(
                "updates per second", i * self.batch_size / elapsed_time,
                self.epoch
            )

            self.save()
            self._update_samplers_weights()
            self.epoch += 1
            start_time = time.time()

    def _start_train_loop(self):
        stored = len(self.replay_buffer)
        while stored < self.start_learning:
            self._fetch_episodes()
            stored = len(self.replay_buffer)
            print(f"waiting for transitions: {stored}/{self.start_learning}")
            time.sleep(1.0)
        self._train()

    def run(self):
        self._update_samplers_weights()
        self._start_redis_loop()
        self._start_train_loop()
