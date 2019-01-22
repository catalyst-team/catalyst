import torch
import torch.nn.functional as F
from catalyst.rl.offpolicy.algorithms.core import Algorithm
from catalyst.rl.offpolicy.algorithms.utils import categorical_loss, \
    quantile_loss


class DDPG(Algorithm):
    """
    Swiss Army knife DDPG algorithm.
    """

    def _init(
        self, values_range=(-10., 10.), critic_distribution=None, **kwargs
    ):
        super()._init(**kwargs)
        self.n_atoms = self.critic.out_features
        self._loss_fn = self._base_loss

        if critic_distribution == "quantile":
            tau_min = 1 / (2 * self.n_atoms)
            tau_max = 1 - tau_min
            tau = torch.linspace(
                start=tau_min, end=tau_max, steps=self.n_atoms
            )
            self.tau = self._to_tensor(tau)
            self._loss_fn = self._quantile_loss
        elif critic_distribution == "categorical":
            self.v_min, self.v_max = values_range
            self.delta_z = (self.v_max - self.v_min) / (self.n_atoms - 1)
            z = torch.linspace(
                start=self.v_min, end=self.v_max, steps=self.n_atoms
            )
            self.z = self._to_tensor(z)
            self._loss_fn = self._categorical_loss

    def _base_loss(self, states_t, actions_t, rewards_t, states_tp1, done_t):
        gamma = self.gamma**self.n_step

        # actor loss
        policy_loss = -torch.mean(self.critic(states_t, self.actor(states_t)))

        # critic loss
        q_values_t = self.critic(states_t, actions_t)
        q_values_tp1 = self.target_critic(
            states_tp1, self.target_actor(states_tp1)
        ).detach()
        q_target_t = rewards_t + (1 - done_t) * gamma * q_values_tp1

        value_loss = self.critic_criterion(q_values_t, q_target_t).mean()

        return policy_loss, value_loss

    def _quantile_loss(
        self, states_t, actions_t, rewards_t, states_tp1, done_t
    ):
        gamma = self.gamma**self.n_step

        # actor loss
        policy_loss = -torch.mean(self.critic(states_t, self.actor(states_t)))

        # critic loss (quantile regression)
        atoms_t = self.critic(states_t, actions_t)
        atoms_tp1 = self.target_critic(
            states_tp1, self.target_actor(states_tp1)
        ).detach()
        atoms_target_t = rewards_t + (1 - done_t) * gamma * atoms_tp1

        value_loss = quantile_loss(
            atoms_t, atoms_target_t, self.tau, self.n_atoms,
            self.critic_criterion
        )

        return policy_loss, value_loss

    def _categorical_loss(
        self, states_t, actions_t, rewards_t, states_tp1, done_t
    ):
        gamma = self.gamma**self.n_step

        # actor loss
        logits_tp0 = self.critic(states_t, self.actor(states_t))
        probs_tp0 = F.softmax(logits_tp0, dim=-1)
        q_values_tp0 = torch.sum(probs_tp0 * self.z, dim=-1)
        policy_loss = -torch.mean(q_values_tp0)

        # critic loss (kl-divergence between categorical distributions)
        logits_t = self.critic(states_t, actions_t)
        logits_tp1 = self.target_critic(
            states_tp1, self.target_actor(states_tp1)
        ).detach()
        atoms_target_t = rewards_t + (1 - done_t) * gamma * self.z

        value_loss = categorical_loss(
            logits_t, logits_tp1, atoms_target_t, self.z, self.delta_z,
            self.v_min, self.v_max
        )

        return policy_loss, value_loss

    def update_step(
        self, policy_loss, value_loss, actor_update=True, critic_update=True
    ):
        # actor update
        actor_update_metrics = {}
        if actor_update:
            actor_update_metrics = self.actor_update(policy_loss) or {}

        # critic update
        critic_update_metrics = {}
        if critic_update:
            critic_update_metrics = self.critic_update(value_loss) or {}

        loss = value_loss + policy_loss
        metrics = {
            "loss": loss.item(),
            "loss_critic": value_loss.item(),
            "loss_actor": policy_loss.item()
        }
        metrics = {**metrics, **actor_update_metrics, **critic_update_metrics}

        return metrics
