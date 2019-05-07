from typing import Union, Dict
import copy
import torch

from catalyst.dl.utils import UtilsFactory
from catalyst.rl.registry import AGENTS
from .utils import soft_update, get_agent_stuff_from_params
from .core import AlgorithmSpec
from catalyst.rl.agents.core import ActorSpec, CriticSpec
from catalyst.rl.environments.core import EnvironmentSpec


class AlgorithmDiscrete(AlgorithmSpec):
    def __init__(
        self,
        critic: CriticSpec,
        gamma: float,
        n_step: int,
        critic_loss_params: Dict = None,
        critic_optimizer_params: Dict = None,
        critic_scheduler_params: Dict = None,
        critic_grad_clip_params: Dict = None,
        critic_tau: float = 1.0,
        **kwargs
    ):
        self._device = UtilsFactory.prepare_device()
        self.critic = critic.to(self._device)
        self.target_critic = copy.deepcopy(critic).to(self._device)

        # preparation
        agent_stuff = get_agent_stuff_from_params(
            agent=self.critic,
            loss_params=critic_loss_params,
            optimizer_params=critic_optimizer_params,
            scheduler_params=critic_scheduler_params,
            grad_clip_params=critic_grad_clip_params
        )
        # criterion
        self._critic_loss_params = agent_stuff["loss_params"]
        self.critic_criterion = agent_stuff["criterion"]
        # optimizer
        self._critic_optimizer_params = agent_stuff["optimizer_params"]
        self.critic_optimizer = agent_stuff["optimizer"]
        # scheduler
        self._critic_scheduler_params = agent_stuff["scheduler_params"]
        self.critic_scheduler = agent_stuff["scheduler"]
        # grad clipping
        self._critic_grad_clip_params = agent_stuff["grad_clip_params"]
        self.critic_grad_clip_fn = agent_stuff["grad_clip_fn"]

        # other hyperparameters
        self._n_step = n_step
        self._gamma = gamma
        self.critic_tau = critic_tau

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

        for key in ["critic"]:
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
        for key in ["critic"]:
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
        raise NotImplementedError()

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
        raise NotImplementedError()

    def target_critic_update(self):
        soft_update(self.target_critic, self.critic, self.critic_tau)

    def update_step(self, value_loss, critic_update=True):
        "updates parameters of neural networks and returns learning metrics"
        raise NotImplementedError

    def train(self, batch, actor_update=False, critic_update=True):
        states_t, actions_t, rewards_t, states_tp1, done_t = \
            batch["state"], batch["action"], batch["reward"], \
            batch["next_state"], batch["done"]

        states_t = self._to_tensor(states_t)
        actions_t = self._to_tensor(actions_t).unsqueeze(1).long()
        rewards_t = self._to_tensor(rewards_t).unsqueeze(1)
        states_tp1 = self._to_tensor(states_tp1)
        done_t = self._to_tensor(done_t).unsqueeze(1)

        """
        states_t: [bs; history_len; observation_len]
        actions_t: [bs; 1]
        rewards_t: [bs; 1]
        states_tp1: [bs; history_len; observation_len]
        done_t: [bs; 1]
        """

        value_loss = self._loss_fn(
            states_t, actions_t, rewards_t, states_tp1, done_t
        )

        metrics = self.update_step(
            value_loss=value_loss,
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
        critic_params = agents_config["critic"]
        critic = AGENTS.get_from_params(
            **critic_params,
            env_spec=env_spec,
        )

        algorithm = cls(
            **config_["algorithm"],
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
        critic_params = agents_config["critic"]
        critic = AGENTS.get_from_params(
            **critic_params,
            env_spec=env_spec,
        )

        return critic
