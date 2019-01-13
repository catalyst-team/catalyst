import copy
import torch

from catalyst.dl.utils import UtilsFactory
from catalyst.rl.agents import AGENTS
from .utils import soft_update


class Algorithm:
    def __init__(
        self,
        actor,
        critic,
        gamma,
        n_step,
        actor_optimizer_params,
        critic_optimizer_params,
        actor_grad_clip_params=None,
        critic_grad_clip_params=None,
        actor_loss_params=None,
        critic_loss_params=None,
        actor_scheduler_params=None,
        critic_scheduler_params=None,
        resume=None,
        load_optimizer=True,
        actor_tau=1.0,
        critic_tau=1.0,
        min_action=-1.0,
        max_action=1.0,
        **kwargs
    ):
        self._device = UtilsFactory.prepare_device()

        self.actor = actor.to(self._device)
        self.critic = critic.to(self._device)

        self.target_actor = copy.deepcopy(actor).to(self._device)
        self.target_critic = copy.deepcopy(critic).to(self._device)

        self.actor_optimizer = UtilsFactory.create_optimizer(
            self.actor, **actor_optimizer_params
        )
        self.critic_optimizer = UtilsFactory.create_optimizer(
            self.critic, **critic_optimizer_params
        )

        self.actor_optimizer_params = actor_optimizer_params
        self.critic_optimizer_params = critic_optimizer_params

        self.actor_scheduler = UtilsFactory.create_scheduler(
            self.actor_optimizer, **actor_scheduler_params
        )
        self.critic_scheduler = UtilsFactory.create_scheduler(
            self.critic_optimizer, **critic_scheduler_params
        )

        self.actor_scheduler_params = actor_scheduler_params
        self.critic_scheduler_params = critic_scheduler_params

        self.n_step = n_step
        self.gamma = gamma

        actor_grad_clip_params = actor_grad_clip_params or {}
        critic_grad_clip_params = critic_grad_clip_params or {}

        self.actor_grad_clip_fn = UtilsFactory.create_grad_clip_fn(
            **actor_grad_clip_params)
        self.critic_grad_clip_fn = UtilsFactory.create_grad_clip_fn(
            **critic_grad_clip_params)

        self.actor_grad_clip_params = actor_grad_clip_params
        self.critic_grad_clip_params = critic_grad_clip_params

        self.actor_criterion = UtilsFactory.create_criterion(
            **(actor_loss_params or {})
        )
        self.critic_criterion = UtilsFactory.create_criterion(
            **(critic_loss_params or {})
        )

        self.actor_loss_params = actor_loss_params
        self.critic_loss_params = critic_loss_params

        self.actor_tau = actor_tau
        self.critic_tau = critic_tau

        self.min_action = min_action
        self.max_action = max_action

        self._init(**kwargs)

        if resume is not None:
            self.load_checkpoint(resume, load_optimizer=load_optimizer)

    def _init(self, **kwards):
        assert len(kwards) == 0

    def __repr__(self):
        str_val = " ".join(
            [
                f"{key}: {str(getattr(self, key, ''))}"
                for key in ["n_step", "gamma", "actor_tau", "critic_tau"]
            ]
        )
        return f"Algorithm. {str_val}"

    def _to_tensor(self, *args, **kwargs):
        return torch.Tensor(*args, **kwargs).to(self._device)

    def train(self, batch, actor_update=True, critic_update=True):
        "returns loss for a batch of transitions"
        raise NotImplementedError

    def get_td_errors(self, batch):
        # @TODO: for prioritized replay
        raise NotImplementedError

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
        soft_update(self.target_actor, self.actor, self.actor_tau)

    def target_critic_update(self):
        soft_update(self.target_critic, self.critic, self.critic_tau)

    def prepare_checkpoint(self):
        checkpoint = {}

        for key in ["actor", "critic"]:
            checkpoint[f"{key}_state_dict"] = getattr(self, key).state_dict()
            for key2 in ["optimizer", "scheduler"]:
                key2 = f"{key}_{key2}"
                value2 = getattr(self, key2, None)
                if value2 is not None:
                    checkpoint[f"{key2}_state_dict"] = value2.state_dict()

        return checkpoint

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

    @classmethod
    def prepare_for_trainer(cls, config):
        config_ = config.copy()

        actor_state_shape = (
            config_["shared"]["history_len"],
            config_["shared"]["state_size"],
        )
        actor_action_size = config_["shared"]["action_size"]
        n_step = config_["shared"]["n_step"]
        gamma = config_["shared"]["gamma"]
        history_len = config_["shared"]["history_len"]
        trainer_state_shape = (config_["shared"]["state_size"], )
        trainer_action_shape = (config_["shared"]["action_size"], )

        actor_fn = config_["actor"].pop("actor", None)
        actor_fn = AGENTS[actor_fn]
        actor = actor_fn.create_from_config(
            state_shape=actor_state_shape,
            action_size=actor_action_size,
            **config_["actor"]
        )

        critic_fn = config_["critic"].pop("critic", None)
        critic_fn = AGENTS[critic_fn]
        critic = critic_fn.create_from_config(
            state_shape=actor_state_shape,
            action_size=actor_action_size,
            **config_["critic"]
        )

        algorithm = cls(
            **config_["algorithm"],
            actor=actor,
            critic=critic,
            n_step=n_step,
            gamma=gamma
        )

        kwargs = {
            "algorithm": algorithm,
            "state_shape": trainer_state_shape,
            "action_shape": trainer_action_shape,
            "n_step": n_step,
            "gamma": gamma,
            "history_len": history_len
        }

        return kwargs

    @classmethod
    def prepare_for_sampler(cls, config):
        config_ = config.copy()

        actor_state_shape = (
            config_["shared"]["history_len"],
            config_["shared"]["state_size"],
        )
        actor_action_size = config_["shared"]["action_size"]

        actor_fn = config_["actor"].pop("actor", None)
        actor_fn = AGENTS[actor_fn]
        actor = actor_fn.create_from_config(
            state_shape=actor_state_shape,
            action_size=actor_action_size,
            **config_["actor"]
        )

        history_len = config_["shared"]["history_len"]

        kwargs = {"actor": actor, "history_len": history_len}

        return kwargs


ALGORITHM = Algorithm
