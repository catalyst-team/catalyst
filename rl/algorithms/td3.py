import copy
import torch
from catalyst.dl.utils import UtilsFactory
import catalyst.rl.agents.agents as agents
from catalyst.rl.algorithms.base import BaseAlgorithm, soft_update


class TD3(BaseAlgorithm):
    def _init(
        self,
        critic2,
        min_action,
        max_action,
        action_noise_std=0.2,
        action_noise_clip=0.5,
        **kwargs
    ):
        super()._init(**kwargs)

        self.min_action = min_action
        self.max_action = max_action
        self.action_noise_std = action_noise_std
        self.action_noise_clip = action_noise_clip

        self.critic2 = critic2.to(self._device)
        self.critic2_optimizer = UtilsFactory.create_optimizer(
            self.critic2, **self.critic_optimizer_params
        )
        self.critic2_scheduler = UtilsFactory.create_scheduler(
            self.critic_optimizer, **self.critic_scheduler_params
        )
        self.target_critic2 = copy.deepcopy(critic2).to(self._device)

    def train(self, batch, actor_update=True, critic_update=True):
        states_t, actions_t, rewards_t, states_tp1, done_t = \
            batch["state"], batch["action"], batch["reward"], \
            batch["next_state"], batch["done"]

        states_t = self.to_tensor(states_t)
        actions_t = self.to_tensor(actions_t)
        rewards_t = self.to_tensor(rewards_t).unsqueeze(1)
        states_tp1 = self.to_tensor(states_tp1)
        done_t = self.to_tensor(done_t).unsqueeze(1)

        # actor loss
        policy_loss = -torch.mean(self.critic(states_t, self.actor(states_t)))

        # critic loss
        actions_tp1 = self.target_actor(states_tp1).detach()
        action_noise = torch.normal(
            mean=torch.zeros_like(actions_tp1), std=self.action_noise_std
        )
        action_noise = action_noise.clamp(
            -self.action_noise_clip, self.action_noise_clip
        )
        actions_tp1 = actions_tp1 + action_noise
        actions_tp1 = actions_tp1.clamp(self.min_action, self.max_action)

        q_values_tp1_1 = self.target_critic(states_tp1, actions_tp1)
        q_values_tp1_2 = self.target_critic2(states_tp1, actions_tp1)
        q_values_tp1 = torch.min(q_values_tp1_1, q_values_tp1_2)

        gamma = self.gamma**self.n_step
        q_target_t = (rewards_t + (1 - done_t) * gamma * q_values_tp1).detach()

        q_values_t_1 = self.critic(states_t, actions_t)
        q_values_t_2 = self.critic2(states_t, actions_t)

        value_loss = self.critic_criterion(q_values_t_1, q_target_t).mean()
        value_loss2 = self.critic_criterion(q_values_t_2, q_target_t).mean()

        metrics = self.update_step(
            policy_loss=policy_loss,
            value_loss=(value_loss, value_loss2),
            actor_update=actor_update,
            critic_update=critic_update
        )

        return metrics

    def update_step(
        self, policy_loss, value_loss, actor_update=True, critic_update=True
    ):

        value_loss, value_loss2 = value_loss

        # actor update
        actor_update_metrics = {}
        if actor_update:
            actor_update_metrics = self.actor_update(policy_loss) or {}

        # critic update
        critic_update_metrics = {}
        if critic_update:
            critic_update_metrics = \
                {
                    **self.critic_update(value_loss),
                    **self.critic2_update(value_loss2)
                } or {}

        loss = value_loss + value_loss2 + policy_loss
        metrics = {
            "loss": loss.item(),
            "loss_critic": value_loss.item(),
            "loss_critic2": value_loss2.item(),
            "loss_actor": policy_loss.item()
        }
        metrics = {**metrics, **actor_update_metrics, **critic_update_metrics}

        return metrics

    def target_critic_update(self):
        soft_update(self.target_critic, self.critic, self.critic_tau)
        soft_update(self.target_critic2, self.critic2, self.critic_tau)

    def critic2_update(self, loss):
        self.critic2.zero_grad()
        self.critic2_optimizer.zero_grad()
        loss.backward()
        if self.critic_grad_clip is not None:
            self.critic_grad_clip(self.critic2.parameters())
        self.critic2_optimizer.step()
        if self.critic2_scheduler is not None:
            self.critic2_scheduler.step()
            return {"lr_critic2": self.critic2_scheduler.get_lr()[0]}

    def load_checkpoint(self, filepath, load_optimizer=True):
        super().load_checkpoint(filepath, load_optimizer)

        checkpoint = UtilsFactory.load_checkpoint(filepath)
        for key in ["critic2"]:
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

    def prepare_checkpoint(self):
        checkpoint = super().prepare_checkpoint()

        for key in ["critic2"]:
            checkpoint[f"{key}_state_dict"] = getattr(self, key).state_dict()
            for key2 in ["optimizer", "scheduler"]:
                key2 = f"{key}_{key2}"
                value2 = getattr(self, key2, None)
                if value2 is not None:
                    checkpoint[f"{key2}_state_dict"] = value2.state_dict()

        return checkpoint


def prepare_for_trainer(config, algo=TD3):
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
    actor_fn = getattr(agents, actor_fn)
    actor = actor_fn(
        state_shape=actor_state_shape,
        action_size=actor_action_size,
        **config_["actor"]
    )

    critic_fn = config_["critic"].pop("critic", None)
    critic_fn = getattr(agents, critic_fn)
    critic = critic_fn(
        state_shape=actor_state_shape,
        action_size=actor_action_size,
        **config_["critic"]
    )
    critic2 = critic_fn(
        state_shape=actor_state_shape,
        action_size=actor_action_size,
        **config_["critic"]
    )

    algorithm = algo(
        **config_["algorithm"],
        actor=actor,
        critic=critic,
        critic2=critic2,
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


def prepare_for_sampler(config):
    config_ = config.copy()

    actor_state_shape = (
        config_["shared"]["history_len"],
        config_["shared"]["state_size"],
    )
    actor_action_size = config_["shared"]["action_size"]

    actor_fn = config_["actor"].pop("actor", None)
    actor_fn = getattr(agents, actor_fn)
    actor = actor_fn(
        **config_["actor"],
        state_shape=actor_state_shape,
        action_size=actor_action_size
    )

    history_len = config_["shared"]["history_len"]

    kwargs = {"actor": actor, "history_len": history_len}

    return kwargs
