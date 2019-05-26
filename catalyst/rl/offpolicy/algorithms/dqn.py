import torch
from catalyst.rl.offpolicy.algorithms.core_discrete import AlgorithmDiscrete
from catalyst.rl.offpolicy.algorithms.utils import categorical_loss, \
    quantile_loss, hyperbolic_gammas


class DQN(AlgorithmDiscrete):

    def _init(self):
        # value distribution approximation
        critic_distribution = self.critic.distribution
        self._loss_fn = self._base_loss
        self._num_heads = self.critic.num_heads
        self._hyperbolic_exponent = self.critic.hyperbolic_exponent
        self._gammas = \
            hyperbolic_gammas(self._gamma, self._hyperbolic_exponent, self._num_heads)
        assert critic_distribution in [None, "categorical", "quantile"]

        if critic_distribution == "categorical":
            self.num_atoms = self.critic.num_atoms
            values_range = self.critic.values_range
            self.v_min, self.v_max = values_range
            self.delta_z = (self.v_max - self.v_min) / (self.num_atoms - 1)
            z = torch.linspace(
                start=self.v_min, end=self.v_max, steps=self.num_atoms
            )
            self.z = self._to_tensor(z)
            self._loss_fn = self._categorical_loss
        elif critic_distribution == "quantile":
            self.num_atoms = self.critic.num_atoms
            tau_min = 1 / (2 * self.num_atoms)
            tau_max = 1 - tau_min
            tau = torch.linspace(
                start=tau_min, end=tau_max, steps=self.num_atoms
            )
            self.tau = self._to_tensor(tau)
            self._loss_fn = self._quantile_loss

    def _base_loss(self, states_t, actions_t, rewards_t, states_tp1, done_t):
        total_loss = 0
        to_list = lambda x: [x] if self._num_heads == 1 else x
        for i, gamma in enumerate(self._gammas):
            gamma = gamma ** self._n_step

            # critic loss
            q_values_t = to_list(self.critic(states_t))[i].squeeze(-1).gather(-1, actions_t)
            q_values_tp1 = \
                to_list(self.target_critic(states_tp1))[i].squeeze(-1).max(-1, keepdim=True)[0]
            q_target_t = rewards_t + (1 - done_t) * gamma * q_values_tp1.detach()
            value_loss = self.critic_criterion(q_values_t, q_target_t).mean()
            total_loss = total_loss + value_loss

        return total_loss

    def _categorical_loss(
        self, states_t, actions_t, rewards_t, states_tp1, done_t
    ):
        total_loss = 0
        to_list = lambda x: [x] if self._num_heads == 1 else x
        for i, gamma in enumerate(self._gammas):
            gamma = gamma ** self._n_step

            # critic loss (kl-divergence between categorical distributions)
            indices_t = actions_t.repeat(1, self.num_atoms).unsqueeze(1)
            logits_t = to_list(self.critic(states_t))[i].gather(1, indices_t).squeeze(1)

            all_logits_tp1 = to_list(self.target_critic(states_tp1))[i].detach()
            q_values_tp1 = torch.sum(
                torch.softmax(all_logits_tp1, dim=-1) * self.z, dim=-1
            )
            actions_tp1 = torch.argmax(q_values_tp1, dim=-1, keepdim=True)
            indices_tp1 = actions_tp1.repeat(1, self.num_atoms).unsqueeze(1)
            logits_tp1 = all_logits_tp1.gather(1, indices_tp1).squeeze(1)
            atoms_target_t = rewards_t + (1 - done_t) * gamma * self.z

            value_loss = categorical_loss(
                logits_t, logits_tp1, atoms_target_t, self.z, self.delta_z,
                self.v_min, self.v_max
            )
            total_loss = total_loss + value_loss

        return total_loss

    def _quantile_loss(
        self, states_t, actions_t, rewards_t, states_tp1, done_t
    ):
        total_loss = 0
        to_list = lambda x: [x] if self._num_heads == 1 else x

        for i, gamma in enumerate(self._gammas):
            gamma = gamma ** self._n_step

            # critic loss (quantile regression)
            indices_t = actions_t.repeat(1, self.num_atoms).unsqueeze(1)
            atoms_t = to_list(self.critic(states_t)).gather(1, indices_t)[i].squeeze(1)

            all_atoms_tp1 = to_list(self.target_critic(states_tp1))[i].detach()
            q_values_tp1 = all_atoms_tp1.mean(dim=-1)
            actions_tp1 = torch.argmax(q_values_tp1, dim=-1, keepdim=True)
            indices_tp1 = actions_tp1.repeat(1, self.num_atoms).unsqueeze(1)
            atoms_tp1 = all_atoms_tp1.gather(1, indices_tp1).squeeze(1)
            atoms_target_t = rewards_t + (1 - done_t) * gamma * atoms_tp1

            value_loss = quantile_loss(
                atoms_t, atoms_target_t, self.tau, self.num_atoms,
                self.critic_criterion
            )
            total_loss = total_loss + value_loss

        return total_loss

    def update_step(self, value_loss, critic_update=True):
        # critic update
        critic_update_metrics = {}
        if critic_update:
            critic_update_metrics = self.critic_update(value_loss) or {}

        loss = value_loss
        metrics = {
            "loss": loss.item(),
            "loss_critic": value_loss.item()
        }
        metrics = {**metrics, **critic_update_metrics}

        return metrics
