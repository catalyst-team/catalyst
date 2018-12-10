import torch
from catalyst.rl.algorithms.utils import quantile_loss
from catalyst.rl.algorithms import TD3, \
    prepare_for_trainer as td3_prepare_for_trainer


class QuantileTD3(TD3):
    def _init(self, **kwargs):
        super()._init(**kwargs)

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
        policy_loss = -torch.mean(self.critic(states_t, self.actor(states_t)))

        # critic loss
        actions_tp1 = self.target_actor(states_tp1).detach()
        action_noise = torch.normal(
            mean=torch.zeros_like(actions_tp1), std=self.action_noise_std
        )
        noise_clip = torch.clamp(
            action_noise, -self.action_noise_clip, self.action_noise_clip
        )
        actions_tp1 = actions_tp1 + noise_clip
        actions_tp1 = actions_tp1.clamp(self.min_action, self.max_action)

        atoms_tp1_1 = self.target_critic(states_tp1, actions_tp1)
        atoms_tp1_2 = self.target_critic2(states_tp1, actions_tp1)

        q_values_tp1_1 = torch.mean(atoms_tp1_1, dim=-1)
        q_values_tp1_2 = torch.mean(atoms_tp1_2, dim=-1)
        q_diff = q_values_tp1_1 - q_values_tp1_2
        mask = q_diff.lt(0).to(torch.float32).detach()[:, None]
        atoms_tp1 = atoms_tp1_1 * mask + atoms_tp1_2 * (1 - mask)

        gamma = self.gamma**self.n_step
        atoms_target_t = (rewards_t +
                          (1 - done_t) * gamma * atoms_tp1).detach()

        atoms_t_1 = self.critic(states_t, actions_t)
        atoms_t_2 = self.critic2(states_t, actions_t)

        value_loss = quantile_loss(
            atoms_t_1,
            atoms_target_t,
            tau=self.tau,
            n_atoms=self.num_atoms,
            criterion=self.critic_criterion
        ).mean()
        value_loss2 = quantile_loss(
            atoms_t_2,
            atoms_target_t,
            tau=self.tau,
            n_atoms=self.num_atoms,
            criterion=self.critic_criterion
        ).mean()

        metrics = self.update_step(
            policy_loss=policy_loss,
            value_loss=(value_loss, value_loss2),
            actor_update=actor_update,
            critic_update=critic_update
        )

        return metrics


def prepare_for_trainer(config):
    return td3_prepare_for_trainer(config, algo=QuantileTD3)
