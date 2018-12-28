import copy
import torch
import torch.nn.functional as F
from catalyst.dl.utils import UtilsFactory
import catalyst.rl.networks.agents as agents
from catalyst.rl.algorithms.base import BaseAlgorithm, soft_update
from catalyst.rl.algorithms.utils import categorical_loss, \
    quantile_loss


class TD3(BaseAlgorithm):
    """
    Swiss Army knife TD3 algorithm.
    """
    def _init(
        self,
        critic2,
        action_noise_std=0.2,
        action_noise_clip=0.5,
        values_range=(-10., 10.),
        critic_distribution=None,
        **kwargs
    ):
        super()._init(**kwargs)
        self.critic_distribution = critic_distribution
        self.num_atoms = self.critic.n_atoms

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

        if critic_distribution == "quantile":
            tau_min = 1 / (2 * self.num_atoms)
            tau_max = 1 - tau_min
            tau = torch.linspace(
                start=tau_min, end=tau_max, steps=self.num_atoms)
            self.tau = self.to_tensor(tau)
        elif critic_distribution == "categorical":
            self.v_min, self.v_max = values_range
            self.delta_z = (self.v_max - self.v_min) / (self.num_atoms - 1)
            z = torch.linspace(
                start=self.v_min, end=self.v_max, steps=self.num_atoms)
            self.z = self.to_tensor(z)

    def train(self, batch, actor_update=True, critic_update=True):
        states_t, actions_t, rewards_t, states_tp1, done_t = \
            batch["state"], batch["action"], batch["reward"], \
            batch["next_state"], batch["done"]

        states_t = self.to_tensor(states_t)
        actions_t = self.to_tensor(actions_t)
        rewards_t = self.to_tensor(rewards_t).unsqueeze(1)
        states_tp1 = self.to_tensor(states_tp1)
        done_t = self.to_tensor(done_t).unsqueeze(1)

        policy_loss, value_loss, value_loss2 = self.calculate_losses(
            states_t, actions_t, rewards_t, states_tp1, done_t
        )

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

    def calculate_losses(
        self, states_t, actions_t, rewards_t, states_tp1, done_t
    ):
        gamma = self.gamma ** self.n_step
        actions_tp1 = self.target_actor(states_tp1).detach()
        action_noise = torch.normal(
            mean=torch.zeros_like(actions_tp1), std=self.action_noise_std
        )
        action_noise = action_noise.clamp(
            -self.action_noise_clip, self.action_noise_clip
        )
        actions_tp1 = actions_tp1 + action_noise
        actions_tp1 = actions_tp1.clamp(self.min_action, self.max_action)

        if self.critic_distribution == "quantile":

            # actor loss
            policy_loss = -torch.mean(
                self.critic(states_t, self.actor(states_t)))

            # critic loss (quantile regression)
            atoms_tp1_1 = self.target_critic(states_tp1, actions_tp1)
            atoms_tp1_2 = self.target_critic2(states_tp1, actions_tp1)
            q_values_tp1_1 = torch.mean(atoms_tp1_1, dim=-1)
            q_values_tp1_2 = torch.mean(atoms_tp1_2, dim=-1)
            q_diff = q_values_tp1_1 - q_values_tp1_2
            mask = q_diff.lt(0).to(torch.float32).detach()[:, None]
            atoms_tp1 = (
                atoms_tp1_1 * mask + atoms_tp1_2 * (1 - mask)).detach()
            atoms_target_t = rewards_t + (1 - done_t) * gamma * atoms_tp1

            atoms_tp1 = self.target_critic(
                states_tp1, self.target_actor(states_tp1)
            ).detach()
            atoms_target_t = rewards_t + (1 - done_t) * gamma * atoms_tp1
            atoms_t_1 = self.critic(states_t, actions_t)
            atoms_t_2 = self.critic2(states_t, actions_t)

            value_loss = quantile_loss(
                atoms_t_1, atoms_target_t,
                self.tau, self.num_atoms, self.critic_criterion)
            value_loss2 = quantile_loss(
                atoms_t_2, atoms_target_t,
                self.tau, self.num_atoms, self.critic_criterion)

        elif self.critic_distribution == "categorical":

            # actor loss
            logits_tp0 = self.critic(states_t, self.actor(states_t))
            probs_tp0 = F.softmax(logits_tp0, dim=-1)
            q_values_tp0 = torch.sum(probs_tp0 * self.z, dim=-1)
            policy_loss = -torch.mean(q_values_tp0)

            # critic loss (kl-divergence between categorical distributions)
            logits_tp1_1 = self.target_critic(states_tp1, actions_tp1)
            logits_tp1_2 = self.target_critic2(states_tp1, actions_tp1)
            probs_tp1_1 = F.softmax(logits_tp1_1, dim=-1)
            probs_tp1_2 = F.softmax(logits_tp1_2, dim=-1)
            q_values_tp1_1 = torch.sum(probs_tp1_1 * self.z, dim=-1)
            q_values_tp1_2 = torch.sum(probs_tp1_2 * self.z, dim=-1)
            q_diff = q_values_tp1_1 - q_values_tp1_2
            mask = q_diff.lt(0).to(torch.float32).detach()[:, None]
            logits_tp1 = (
                logits_tp1_1 * mask + logits_tp1_2 * (1 - mask)).detach()

            logits_t_1 = self.critic(states_t, actions_t)
            logits_t_2 = self.critic2(states_t, actions_t)
            atoms_target_t = rewards_t + (1 - done_t) * gamma * self.z

            value_loss = categorical_loss(
                logits_t_1, logits_tp1, atoms_target_t,
                self.z, self.delta_z, self.v_min, self.v_max)
            value_loss2 = categorical_loss(
                logits_t_2, logits_tp1, atoms_target_t,
                self.z, self.delta_z, self.v_min, self.v_max)

        else:

            # actor loss
            policy_loss = -torch.mean(
                self.critic(states_t, self.actor(states_t)))

            # critic loss
            q_values_t_1 = self.critic(states_t, actions_t)
            q_values_t_2 = self.critic2(states_t, actions_t)
            q_values_tp1_1 = self.target_critic(states_tp1, actions_tp1)
            q_values_tp1_2 = self.target_critic2(states_tp1, actions_tp1)
            q_values_tp1 = torch.min(q_values_tp1_1, q_values_tp1_2).detach()
            q_target_t = rewards_t + (1 - done_t) * gamma * q_values_tp1

            value_loss = self.critic_criterion(
                q_values_t_1, q_target_t).mean()
            value_loss2 = self.critic_criterion(
                q_values_t_2, q_target_t).mean()

        return policy_loss, value_loss, value_loss2

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
