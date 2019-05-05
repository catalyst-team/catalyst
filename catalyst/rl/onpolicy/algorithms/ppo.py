from typing import Union, Dict

import numpy as np
import torch

from catalyst.dl.utils import UtilsFactory
from catalyst.rl.registry import AGENTS
from catalyst.rl.offpolicy.algorithms.utils import get_agent_stuff_from_params
from catalyst.rl.offpolicy.algorithms.core import AlgorithmSpec
from catalyst.rl.agents.core import ActorSpec, CriticSpec
from catalyst.rl.environments.core import EnvironmentSpec


def create_gamma_matrix(tau, matrix_size):
    """
    Matrix of the following form
    --------------------
    1     y   y^2    y^3
    0     1     y    y^2
    0     0     1      y
    0     0     0      1
    --------------------
    for fast gae calculation
    """
    i = np.arange(matrix_size)
    j = np.arange(matrix_size)
    pow_ = i[None, :] - j[:, None]
    mat = np.power(tau, pow_) * (pow_ >= 0)
    return mat


class PPO(AlgorithmSpec):

    def __init__(
        self,
        actor: ActorSpec,
        critic: CriticSpec,
        gamma: float,
        n_step: int,
        actor_loss_params: Dict = None,
        critic_loss_params: Dict = None,
        actor_optimizer_params: Dict = None,
        critic_optimizer_params: Dict = None,
        actor_scheduler_params: Dict = None,
        critic_scheduler_params: Dict = None,
        actor_grad_clip_params: Dict = None,
        critic_grad_clip_params: Dict = None,
        **kwargs
    ):
        self._device = UtilsFactory.prepare_device()

        self.actor = actor.to(self._device)
        self.critic = critic.to(self._device)

        # actor preparation
        actor_stuff = get_agent_stuff_from_params(
            agent=self.actor,
            loss_params=actor_loss_params,
            optimizer_params=actor_optimizer_params,
            scheduler_params=actor_scheduler_params,
            grad_clip_params=actor_grad_clip_params
        )
        # criterion
        self._actor_loss_params = actor_stuff["loss_params"]
        self.actor_criterion = actor_stuff["criterion"]
        # optimizer
        self._actor_optimizer_params = actor_stuff["optimizer_params"]
        self.actor_optimizer = actor_stuff["optimizer"]
        # scheduler
        self._actor_scheduler_params = actor_stuff["scheduler_params"]
        self.actor_scheduler = actor_stuff["scheduler"]
        # grad clipping
        self._actor_grad_clip_params = actor_stuff["grad_clip_params"]
        self.actor_grad_clip_fn = actor_stuff["grad_clip_fn"]

        # critic preparation
        critic_stuff = get_agent_stuff_from_params(
            agent=self.critic,
            loss_params=critic_loss_params,
            optimizer_params=critic_optimizer_params,
            scheduler_params=critic_scheduler_params,
            grad_clip_params=critic_grad_clip_params
        )
        # criterion
        self._critic_loss_params = critic_stuff["loss_params"]
        self.critic_criterion = critic_stuff["criterion"]
        # optimizer
        self._critic_optimizer_params = critic_stuff["optimizer_params"]
        self.critic_optimizer = critic_stuff["optimizer"]
        # scheduler
        self._critic_scheduler_params = critic_stuff["scheduler_params"]
        self.critic_scheduler = critic_stuff["scheduler"]
        # grad clipping
        self._critic_grad_clip_params = critic_stuff["grad_clip_params"]
        self.critic_grad_clip_fn = critic_stuff["grad_clip_fn"]

        # other hyperparameters
        self._n_step = n_step
        self._gamma = gamma

        # other init
        self._init(**kwargs)

    def _init(
        self,
        use_value_clipping=True,
        gae_lambda=0.95,
        clip_eps=0.2,
        max_episode_length=1000,
        entropy_reg_coefficient=0.,
        **kwargs
    ):
        assert len(kwargs) == 0

        self.use_value_clipping = use_value_clipping
        self.clip_eps = clip_eps
        self.entropy_reg_coefficient = entropy_reg_coefficient

        # matrix for estimating advantages with GAE
        # used in policy loss
        self.gam_lam_matrix = create_gamma_matrix(
            self.gamma * gae_lambda, max_episode_length)

        # matrix for estimating cummulative discounted returns
        # used in value loss
        self.gam_matrix = create_gamma_matrix(
            self.gamma, max_episode_length)

    @property
    def n_step(self) -> int:
        return self._n_step

    @property
    def gamma(self) -> float:
        return self._gamma

    def _to_tensor(self, *args, **kwargs):
        return torch.Tensor(*args, **kwargs).to(self._device)

    def pack_checkpoint(self):
        checkpoint = {}

        for key in ["actor", "critic"]:
            checkpoint[f"{key}_state_dict"] = getattr(self, key).state_dict()
            for key2 in ["optimizer", "scheduler"]:
                key2 = f"{key}_{key2}"
                value2 = getattr(self, key2, None)
                if value2 is not None:
                    checkpoint[f"{key2}_state_dict"] = value2.state_dict()

    def unpack_checkpoint(self, checkpoint):
        raise NotImplementedError()

    def save_checkpoint(self, filepath):
        raise NotImplementedError()

    def load_checkpoint(self, filepath, load_optimizer=True):
        checkpoint = UtilsFactory.load_checkpoint(filepath)
        for key in ["actor", "critic"]:
            value_l = getattr(self, key, None)
            if value_l is not None:
                value_r = checkpoint[f"{key}_state_dict"]
                value_l.load_state_dict(value_r)

            if load_optimizer:
                for key2 in ["optimizer", "scheduler"]:
                    key2 = f"{key}_{key2}"
                    value_l = getattr(self, key2, None)
                    if value_l is not None:
                        value_r = checkpoint[f"{key2}_state_dict"]
                        value_l.load_state_dict(value_r)

    def actor_update(self, loss):
        self.actor.zero_grad()
        self.actor_optimizer.zero_grad()
        loss.backward()
        if self.actor_grad_clip_fn is not None:
            self.actor_grad_clip_fn(self.actor.parameters())
        self.actor_optimizer.step()
        if self.actor_scheduler is not None:
            self.actor_scheduler.step()
            return {"lr_actor": self.actor_scheduler.get_lr()[0]}

    def critic_update(self, loss):
        self.critic.zero_grad()
        self.critic_optimizer.zero_grad()
        loss.backward()
        if self.critic_grad_clip_fn is not None:
            self.critic_grad_clip_fn(self.critic.parameters())
        self.critic_optimizer.step()
        if self.critic_scheduler is not None:
            self.critic_scheduler.step()
            return {"lr_critic": self.critic_scheduler.get_lr()[0]}

    def target_actor_update(self):
        pass

    def target_critic_update(self):
        pass

    def train(self, batch, actor_update=True, critic_update=True):
        states, actions, returns, values, advantages, log_pis = \
            batch["state"], batch["action"], batch["return"], \
            batch["value"], batch["advantage"], batch["log_pi"]

        states = self._to_tensor(states)
        actions = self._to_tensor(actions)
        returns = self._to_tensor(returns)
        old_values = self._to_tensor(values)
        advantages = self._to_tensor(advantages)
        old_log_pi = self._to_tensor(log_pis)

        values_t = self.critic(states).squeeze()
        if self.use_value_clipping:
            values_clip = old_values + torch.clamp(
                values_t - old_values, -self.clip_eps, self.clip_eps)
            val_loss1 = (values_t - returns).pow(2)
            val_loss2 = (values_clip - returns).pow(2)
            value_loss = 0.5 * torch.max(val_loss1, val_loss2).mean()
        else:
            value_loss = self.critic_criterion(
                values_t[:, None], returns[:, None]).mean()

        # actor loss
        _, dist = self.actor(states, with_log_pi=True)
        log_pi = dist.log_prob(actions)

        ratio = torch.exp(log_pi - old_log_pi)
        surr1 = advantages * ratio
        surr2 = advantages * torch.clamp(
            ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps)
        policy_loss = -torch.min(surr1, surr2).mean()

        entropy = -(torch.exp(log_pi) * log_pi).mean()
        entropy_reg = self.entropy_reg_coefficient * entropy
        policy_loss = policy_loss + entropy_reg

        # actor update
        actor_update_metrics = self.actor_update(policy_loss) or {}

        # critic update
        critic_update_metrics = self.critic_update(value_loss) or {}

        metrics = {
            "loss_actor": policy_loss.item(),
            "loss_critic": value_loss.item()
        }
        metrics = {**metrics, **actor_update_metrics, **critic_update_metrics}
        return metrics

    def evaluate_trajectory(self, states, actions, rewards):
        states = self._to_tensor(states)
        actions = self._to_tensor(actions)
        rewards = np.array(rewards)

        ep_len = rewards.shape[0]
        values = torch.zeros((ep_len + 1, 1)).to(self._device)
        values[:ep_len] = self.critic(states)
        values = values.detach().cpu().numpy().reshape(-1)

        _, dist = self.actor(states, with_log_pi=True)
        log_pis = dist.log_prob(actions)
        log_pis = log_pis.detach().cpu().numpy().reshape(-1)

        deltas = rewards + self.gamma * values[1:] - values[:-1]
        advantages = np.dot(self.gam_lam_matrix[:ep_len, :ep_len], deltas)
        returns = np.dot(self.gam_matrix[:ep_len, :ep_len], rewards)

        return [returns, values[:ep_len], advantages, log_pis]

    @classmethod
    def prepare_for_trainer(
            cls,
            env_spec: EnvironmentSpec,
            config: Dict
    ) -> "AlgorithmSpec":
        config_ = config.copy()
        agents_config = config_["agents"]

        actor_params = agents_config["actor"]
        actor = AGENTS.get_from_params(
            **actor_params,
            env_spec=env_spec,
        )

        critic_params = agents_config["critic"]
        critic = AGENTS.get_from_params(
            **critic_params,
            env_spec=env_spec,
        )

        algorithm = cls(
            **config_["algorithm"],
            actor=actor,
            critic=critic,
        )

        return algorithm

    @classmethod
    def prepare_for_sampler(
            cls,
            env_spec: EnvironmentSpec,
            config: Dict
    ) -> Union[ActorSpec, CriticSpec]:
        config_ = config.copy()
        agents_config = config_["agents"]
        actor_params = agents_config["actor"]
        actor = AGENTS.get_from_params(
            **actor_params,
            env_spec=env_spec,
        )

        return actor
