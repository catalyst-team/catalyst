import copy
import torch
from catalyst.rl.algorithms.utils import quantile_loss
from catalyst.utils.factory import UtilsFactory
import catalyst.rl.networks.agents as agents
from catalyst.rl.algorithms import TD3
from catalyst.rl.algorithms.base import soft_update


class QuantileTD3WithValue(TD3):
    def _init(self, critic_v, **kwargs):
        super()._init(**kwargs)

        self.critic_v = critic_v.to(self._device)
        self.critic_v_optimizer = UtilsFactory.create_optimizer(
            self.critic_v, **self.critic_optimizer_params)
        self.critic_v_scheduler = UtilsFactory.create_scheduler(
            self.critic_v_optimizer, **self.critic_scheduler_params)
        self.target_critic_v = copy.deepcopy(critic_v).to(self._device)

        self.num_atoms = self.critic.n_atoms
        tau_min = 1 / (2 * self.num_atoms)
        tau_max = 1 - tau_min
        tau = torch.linspace(start=tau_min, end=tau_max, steps=self.num_atoms)
        self.tau = self.to_tensor(tau)

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
        policy_loss = -torch.mean(
            self.critic(states_t, self.actor(states_t)))

        # critic loss
        actions_tp0 = self.actor(states_t).detach()
        action_noise = torch.normal(
            mean=torch.zeros_like(actions_tp0), std=self.action_noise_std)
        noise_clip = torch.clamp(
            action_noise, -self.action_noise_clip, self.action_noise_clip)
        actions_tp0 = actions_tp0 + noise_clip
        actions_tp0 = actions_tp0.clamp(self.min_action, self.max_action)

        atoms_tp0_1 = self.target_critic(states_t, actions_tp0)
        atoms_tp0_2 = self.target_critic2(states_t, actions_tp0)

        q_values_tp0_1 = torch.mean(atoms_tp0_1, dim=-1)
        q_values_tp0_2 = torch.mean(atoms_tp0_2, dim=-1)
        q_diff = q_values_tp0_1 - q_values_tp0_2
        mask = q_diff.lt(0).to(torch.float32).detach()[:, None]
        atoms_tp0 = atoms_tp0_1 * mask + atoms_tp0_2 * (1 - mask)

        v_atoms_t = self.critic_v(states_t)
        v_value_loss = quantile_loss(
            v_atoms_t, atoms_tp0.detach(),
            tau=self.tau, n_atoms=self.num_atoms,
            criterion=self.critic_criterion).mean()

        v_atoms_tp1 = self.target_critic_v(states_tp1)
        gamma = self.gamma ** self.n_step
        atoms_target_t = (
                    rewards_t + (1 - done_t) * gamma * v_atoms_tp1).detach()
        atoms_t_1 = self.critic(states_t, actions_t)
        atoms_t_2 = self.critic2(states_t, actions_t)
        value_loss = quantile_loss(
            atoms_t_1, atoms_target_t,
            tau=self.tau, n_atoms=self.num_atoms,
            criterion=self.critic_criterion).mean()
        value_loss2 = quantile_loss(
            atoms_t_2, atoms_target_t,
            tau=self.tau, n_atoms=self.num_atoms,
            criterion=self.critic_criterion).mean()

        metrics = self.update_step(
            policy_loss=policy_loss,
            value_loss=(v_value_loss, value_loss, value_loss2),
            actor_update=actor_update,
            critic_update=critic_update)

        return metrics

    def update_step(
            self, policy_loss, value_loss,
            actor_update=True, critic_update=True):

        v_value_loss, value_loss, value_loss2 = value_loss

        # actor update
        actor_update_metrics = {}
        if actor_update:
            actor_update_metrics = self.actor_update(policy_loss) or {}

        # critic update
        critic_update_metrics = {}
        if critic_update:
            critic_update_metrics = \
                {
                    **self.critic_v_update(v_value_loss),
                    **self.critic_update(value_loss),
                    **self.critic2_update(value_loss2)
                } or {}

        loss = v_value_loss + value_loss + value_loss2 + policy_loss
        metrics = {
            "loss": loss.item(),
            "loss_vritic_v": v_value_loss.item(),
            "loss_critic": value_loss.item(),
            "loss_critic2": value_loss2.item(),
            "loss_actor": policy_loss.item()
        }
        metrics = {**metrics, **actor_update_metrics, **critic_update_metrics}

        return metrics

    def target_critic_update(self):
        soft_update(
            self.target_critic_v, self.critic_v,
            self.critic_tau)
        soft_update(
            self.target_critic, self.critic,
            self.critic_tau)
        soft_update(
            self.target_critic2, self.critic2,
            self.critic_tau)

    def critic_v_update(self, loss):
        self.critic_v.zero_grad()
        self.critic_v_optimizer.zero_grad()
        loss.backward()
        if self.critic_grad_clip is not None:
            self.critic_grad_clip(self.critic_v.parameters())
        self.critic_v_optimizer.step()
        if self.critic_v_scheduler is not None:
            self.critic_v_scheduler.step()
            return {"lr_critic_v": self.critic_v_scheduler.get_lr()[0]}

    def load_checkpoint(self, filepath):
        super().load_checkpoint(filepath)
        checkpoint = UtilsFactory.load_checkpoint(filepath)
        for key in ["critic_v"]:
            value_l = getattr(self, key, None)
            if value_l is not None:
                value_r = checkpoint[f"{key}_state_dict"]
                value_l.load_state_dict(value_r)

            for key2 in ["optimizer", "scheduler"]:
                key2 = f"{key}_{key2}"
                value_l = getattr(self, key2, None)
                if value_l is not None:
                    value_r = checkpoint[f"{key2}_state_dict"]
                    value_l.load_state_dict(value_r)

    def prepare_checkpoint(self):
        checkpoint = super().prepare_checkpoint()

        for key in ["critic_v"]:
            checkpoint[f"{key}_state_dict"] = getattr(self, key).state_dict()
            for key2 in ["optimizer", "scheduler"]:
                key2 = f"{key}_{key2}"
                value2 = getattr(self, key2, None)
                if value2 is not None:
                    checkpoint[f"{key2}_state_dict"] = value2.state_dict()

        return checkpoint


def prepare_for_trainer(config, algo=QuantileTD3WithValue):
    config_ = config.copy()

    actor_state_shape = (
        config_["shared"]["history_len"],
        config_["shared"]["state_size"],)
    actor_action_size = config_["shared"]["action_size"]
    n_step = config_["shared"]["n_step"]
    gamma = config_["shared"]["gamma"]
    history_len = config_["shared"]["history_len"]
    trainer_state_shape = (config_["shared"]["state_size"],)
    trainer_action_shape = (config_["shared"]["action_size"],)

    actor_fn = config_["actor"].pop("actor", None)
    actor_fn = getattr(agents, actor_fn)
    actor = actor_fn(
        state_shape=actor_state_shape,
        action_size=actor_action_size,
        **config_["actor"])

    critic_fn = config_["critic"].pop("critic", None)
    critic_fn = getattr(agents, critic_fn)

    critic_v_fn = config_["critic"].pop("value_critic", None)
    critic_v_fn = getattr(agents, critic_v_fn)

    critic = critic_fn(
        state_shape=actor_state_shape,
        action_size=actor_action_size,
        **config_["critic"])
    critic2 = critic_fn(
        state_shape=actor_state_shape,
        action_size=actor_action_size,
        **config_["critic"])

    critic_v = critic_v_fn(
        state_shape=actor_state_shape,
        **config_["critic"])

    algorithm = algo(
        **config_["algorithm"],
        actor=actor,
        critic_v=critic_v, critic=critic, critic2=critic2,
        n_step=n_step, gamma=gamma)

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
        config_["shared"]["state_size"],)
    actor_action_size = config_["shared"]["action_size"]

    actor_fn = config_["actor"].pop("actor", None)
    actor_fn = getattr(agents, actor_fn)
    actor = actor_fn(
        **config_["actor"],
        state_shape=actor_state_shape,
        action_size=actor_action_size)

    history_len = config_["shared"]["history_len"]

    kwargs = {
        "actor": actor,
        "history_len": history_len
    }

    return kwargs
