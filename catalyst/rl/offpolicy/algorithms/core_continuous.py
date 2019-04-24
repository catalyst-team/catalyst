from typing import Union, Dict
import copy
from gym.spaces import Box
import torch

from catalyst.dl.utils import UtilsFactory
from catalyst.rl.registry import AGENTS
from .utils import soft_update, get_agent_stuff_from_params
from .core import AlgorithmSpec
from catalyst.rl.agents.core import ActorSpec, CriticSpec
from catalyst.rl.environments.core import EnvironmentSpec


class AlgorithmContinuous(AlgorithmSpec):
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
        actor_tau=1.0,
        critic_tau=1.0,
        action_boundaries=None,
        **kwargs
    ):
        self._device = UtilsFactory.prepare_device()

        self.actor = actor.to(self._device)
        self.critic = critic.to(self._device)

        self.target_actor = copy.deepcopy(actor).to(self._device)
        self.target_critic = copy.deepcopy(critic).to(self._device)

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
        self._actor_tau = actor_tau
        self._critic_tau = critic_tau

        if action_boundaries is not None:
            assert len(action_boundaries) == 2, \
                "Should be min and max action boundaries"
            self._action_boundaries = action_boundaries

        # other init
        self._init(**kwargs)

    def _init(self, **kwargs):
        assert len(kwargs) == 0

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

        return checkpoint

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
        soft_update(self.target_actor, self.actor, self._actor_tau)

    def target_critic_update(self):
        soft_update(self.target_critic, self.critic, self._critic_tau)

    def update_step(
        self, policy_loss, value_loss, actor_update=True, critic_update=True
    ):
        "updates parameters of neural networks and returns learning metrics"
        raise NotImplementedError

    def train(self, batch, actor_update=True, critic_update=True):
        states_t, actions_t, rewards_t, states_tp1, done_t = \
            batch["state"], batch["action"], batch["reward"], \
            batch["next_state"], batch["done"]

        states_t = self._to_tensor(states_t)
        actions_t = self._to_tensor(actions_t)
        rewards_t = self._to_tensor(rewards_t).unsqueeze(1)
        states_tp1 = self._to_tensor(states_tp1)
        done_t = self._to_tensor(done_t).unsqueeze(1)

        """
        states_t: [bs; history_len; observation_len]
        actions_t: [bs; action_len]
        rewards_t: [bs; 1]
        states_tp1: [bs; history_len; observation_len]
        done_t: [bs; 1]
        """

        policy_loss, value_loss = self._loss_fn(
            states_t, actions_t, rewards_t, states_tp1, done_t
        )

        metrics = self.update_step(
            policy_loss=policy_loss,
            value_loss=value_loss,
            actor_update=actor_update,
            critic_update=critic_update
        )

        return metrics

    def get_td_errors(self, batch):
        # @TODO: for prioritized replay
        raise NotImplementedError

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

        action_space = env_spec.action_space
        assert isinstance(action_space, Box)
        action_boundaries = [
            action_space.low[0],
            action_space.high[0]
        ]

        algorithm = cls(
            **config_["algorithm"],
            action_boundaries=action_boundaries,
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
