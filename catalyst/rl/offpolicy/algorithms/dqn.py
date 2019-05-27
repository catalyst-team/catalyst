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
        self._hyperbolic_constant = self.critic.hyperbolic_constant
        self._gammas = \
            hyperbolic_gammas(
                self._gamma,
                self._hyperbolic_constant,
                self._num_heads
            )
        self._gammas = torch.Tensor(self._gammas).to(self._device)
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

        # If we have only one head, unsqueeze its output for coherence
        if self._num_heads == 1:
            self.critic_fn = lambda x: self.critic(x).unsqueeze(1)
            self.target_critic_fn = \
                lambda x: self.target_critic(x).unsqueeze(1)
        else:
            self.critic_fn = lambda x: self.critic(x)
            self.target_critic_fn = lambda x: self.target_critic(x)

    def _base_loss(self, states_t, actions_t, rewards_t, states_tp1, done_t):

        # Array of size [num_heads,]
        gammas = self._gammas ** self._n_step
        critic = self.critic_fn
        target_critic = self.target_critic_fn

        # We use the same done_t, rewards_t, actions_t for each head
        done_t = done_t[:, None, :]
        rewards_t = rewards_t[:, None, :]
        actions_t = actions_t.unsqueeze(1).repeat(1, self._num_heads, 1)
        gammas = gammas[None, :, None]

        q_values_t = critic(states_t).squeeze(-1).gather(-1, actions_t)
        q_values_tp1 = \
            target_critic(states_tp1).squeeze(-1).max(-1, keepdim=True)[0]
        q_target_t = rewards_t + (1 - done_t * gammas * q_values_tp1.detach())
        value_loss = self.critic_criterion(q_values_t, q_target_t).mean()

        return value_loss

    def _categorical_loss(
        self, states_t, actions_t, rewards_t, states_tp1, done_t
    ):

        critic = self.critic_fn
        target_critic = self.target_critic_fn
        gammas = (self._gammas ** self._n_step)[None, :, None]
        done_t = done_t[:, None, :]
        rewards_t = rewards_t[:, None, :]
        actions_t = actions_t[:, None, None, :]
        indices_t = actions_t.repeat(1, self._num_heads, 1, self.num_atoms)

        logits_t = critic(states_t).gather(-2, indices_t).squeeze(-2)

        all_logits_tp1 = target_critic(states_tp1).detach()
        q_values_tp1 = torch.sum(
            torch.softmax(all_logits_tp1, dim=-1) * self.z, dim=-1
        )
        actions_tp1 = torch.argmax(q_values_tp1, dim=-1, keepdim=True)
        indices_tp1 = \
            actions_tp1.unsqueeze(-1).repeat(1, 1, 1, self.num_atoms)
        logits_tp1 = all_logits_tp1.gather(-2, indices_tp1).squeeze(-2)
        atoms_target_t = rewards_t + (1 - done_t) * gammas * self.z

        value_loss = categorical_loss(
            logits_t.view(-1, self.num_atoms),
            logits_tp1.view(-1, self.num_atoms),
            atoms_target_t.view(-1, self.num_atoms), self.z,
            self.delta_z,
            self.v_min, self.v_max
        )

        return value_loss

    def _quantile_loss(
        self, states_t, actions_t, rewards_t, states_tp1, done_t
    ):

        critic = self.critic_fn
        target_critic = self.target_critic_fn
        gammas = (self._gammas ** self._n_step)[None, :, None]
        done_t = done_t[:, None, :]
        rewards_t = rewards_t[:, None, :]
        actions_t = actions_t[:, None, None, :]
        indices_t = actions_t.repeat(1, self._num_heads, 1, self.num_atoms)

        # critic loss (quantile regression)

        atoms_t = critic(states_t).gather(-2, indices_t).squeeze(-2)

        all_atoms_tp1 = target_critic(states_tp1).detach()
        q_values_tp1 = all_atoms_tp1.mean(dim=-1)
        actions_tp1 = torch.argmax(q_values_tp1, dim=-1, keepdim=True)
        indices_tp1 = actions_tp1.unsqueeze(-1).repeat(1, 1, 1, self.num_atoms)
        atoms_tp1 = all_atoms_tp1.gather(-2, indices_tp1).squeeze(-2)
        atoms_target_t = rewards_t + (1 - done_t) * gammas * atoms_tp1

        value_loss = quantile_loss(
            atoms_t.view(-1, self.num_atoms),
            atoms_target_t.view(-1, self.num_atoms),
            self.tau, self.num_atoms,
            self.critic_criterion
        )

        return value_loss

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
