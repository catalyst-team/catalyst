from typing import Any, Dict, List, Mapping

import numpy as np

import torch

from catalyst.core import _Runner
from catalyst.rl2 import (
    AlgorithmSpec, EnvironmentSpec, RLExperiment, RLState, utils
)

# RLRunner has only one stage (?) - endless training
# each Epoch we recalculate training loader based on current Replay buffer
# then -> typical training on loader with selected algorithm
#


class RLRunner(_Runner):
    experiment: RLExperiment
    state: RLState

    def _fetch_rollouts(self):
        pass

    def _prepare_for_stage(self, stage: str):
        super()._prepare_for_stage(stage=stage)

        self.algorithm: AlgorithmSpec = \
            self.experiment.get_algorithm(stage=stage)
        self.environment: EnvironmentSpec = \
            self.experiment.get_environment(stage=stage)

    def _prepare_for_epoch(self, stage: str, epoch: int):
        super()._prepare_for_epoch(stage=stage, epoch=epoch)

        # @TODO: remove this trick
        utils.set_global_seed(self.experiment.initial_seed + epoch)
        loaders = self.experiment.get_loaders(stage=stage, epoch=epoch)
        self.loaders = loaders

    def _run_batch_train_step(self, batch: Mapping[str, Any]):
        # todo: should implement different training steps
        #  for different algorithms
        metrics: Dict = self.algorithm.train_on_batch(
            batch,
            actor_update=(self.state.step % self.state.actor_grad_period == 0),
            critic_update=(
                    self.state.step % self.state.critic_grad_period == 0
            ),
        ) or {}

        metrics_ = self._update_target_weights(self.state.step) or {}
        metrics.update(**metrics_)
        self.state.metric_manager.add_batch_value(metrics_dict=metrics)

    def forward(self, batch: Mapping[str, Any], **kwargs) -> Mapping[str, Any]:
        # todo: should implement different policy -> action
        #  for different use-cases: discrete, continuous action spaces
        pass

    @torch.no_grad()
    def inference(
        self,
        sampler_ids: List[int],
        run_ids: List[int],
        states: np.ndarray,
        rewards: np.ndarray,
    ):
        # looks like production-ready thing
        # @TODO: make a microservice from this method
        batch = None
        actions = self.predict_batch(batch)
        return actions
