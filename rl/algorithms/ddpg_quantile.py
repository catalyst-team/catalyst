import torch
from catalyst.rl.algorithms.utils import quantile_loss
from catalyst.rl.algorithms.base import \
    prepare_for_trainer as base_prepare_for_trainer
from catalyst.rl.algorithms.ddpg import DDPG


class QuantileDDPG(DDPG):
    def _init(self, **kwargs):
        super()._init(**kwargs)

        num_atoms = self.critic.n_atoms
        tau_min = 1 / (2 * num_atoms)
        tau_max = 1 - tau_min
        tau = torch.linspace(start=tau_min, end=tau_max, steps=num_atoms)
        self.tau = self.to_tensor(tau)
        self.num_atoms = num_atoms

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
        atoms_t = self.critic(states_t, actions_t)
        atoms_tp1 = self.target_critic(
            states_tp1,
            self.target_actor(states_tp1)).detach()

        gamma = self.gamma ** self.n_step
        atoms_target_t = (
                rewards_t + (1 - done_t) * gamma * atoms_tp1).detach()

        value_loss = quantile_loss(
            atoms_t, atoms_target_t,
            tau=self.tau, n_atoms=self.num_atoms,
            criterion=self.critic_criterion).mean()

        metrics = self.update_step(
            policy_loss=policy_loss,
            value_loss=value_loss,
            actor_update=actor_update,
            critic_update=critic_update)

        return metrics


def prepare_for_trainer(config):
    return base_prepare_for_trainer(config, algo=QuantileDDPG)
