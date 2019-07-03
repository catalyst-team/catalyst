import torch
from .critic import OffpolicyCritic
from catalyst.rl import utils


class DQN(OffpolicyCritic):

    def _init(self):
        # value distribution approximation
        critic_distribution = self.critic.distribution
        self._loss_fn = self._base_loss
        self._num_heads = self.critic.num_heads
        self._hyperbolic_constant = self.critic.hyperbolic_constant
        self._gammas = \
            utils.hyperbolic_gammas(
                self._gamma,
                self._hyperbolic_constant,
                self._num_heads
            )
        self._gammas = utils.any2device(self._gammas, device=self._device)
        assert critic_distribution in [None, "categorical", "quantile"]

        if critic_distribution == "categorical":
            self.num_atoms = self.critic.num_atoms
            values_range = self.critic.values_range
            self.v_min, self.v_max = values_range
            self.delta_z = (self.v_max - self.v_min) / (self.num_atoms - 1)
            z = torch.linspace(
                start=self.v_min, end=self.v_max, steps=self.num_atoms
            )
            self.z = utils.any2device(z, device=self._device)
            self._loss_fn = self._categorical_loss
        elif critic_distribution == "quantile":
            self.num_atoms = self.critic.num_atoms
            tau_min = 1 / (2 * self.num_atoms)
            tau_max = 1 - tau_min
            tau = torch.linspace(
                start=tau_min, end=tau_max, steps=self.num_atoms
            )
            self.tau = utils.any2device(tau, device=self._device)
            self._loss_fn = self._quantile_loss
        else:
            assert self.critic_criterion is not None

    def _base_loss(self, states_t, actions_t, rewards_t, states_tp1, done_t):

        # Array of size [num_heads,]
        gammas = self._gammas ** self._n_step

        # We use the same done_t, rewards_t, actions_t for each head
        done_t = done_t[:, None, :]
        # B x 1 x 1

        rewards_t = rewards_t[:, None, :]
        # B x 1 x 1

        actions_t = actions_t.unsqueeze(1).repeat(1, self._num_heads, 1)
        # B x num_heads x 1

        gammas = gammas[None, :, None]
        # 1 x num_heads x 1

        q_values_t = self.critic(states_t).squeeze(-1).gather(-1, actions_t)
        # B x num_heads x 1

        q_values_tp1 = \
            self.target_critic(states_tp1).squeeze(-1).max(-1, keepdim=True)[0]
        # B x num_heads x 1
        q_target_t = rewards_t + (1 - done_t) * gammas * q_values_tp1.detach()
        value_loss = self.critic_criterion(q_values_t, q_target_t).mean()

        return value_loss

    def _categorical_loss(
        self, states_t, actions_t, rewards_t, states_tp1, done_t
    ):

        gammas = (self._gammas ** self._n_step)[None, :, None]
        # 1 x num_heads x 1

        done_t = done_t[:, None, :]  # B x 1 x 1
        rewards_t = rewards_t[:, None, :]  # B x 1 x 1
        actions_t = actions_t[:, None, None, :]  # B x 1 x 1 x 1
        indices_t = actions_t.repeat(1, self._num_heads, 1, self.num_atoms)
        # B x num_heads x 1 x num_atoms

        logits_t = self.critic(states_t).gather(-2, indices_t).squeeze(-2)
        # B x num_heads x num_atoms

        all_logits_tp1 = self.target_critic(states_tp1).detach()
        # B x num_heads x num_actions x num_atoms

        q_values_tp1 = torch.sum(
            torch.softmax(all_logits_tp1, dim=-1) * self.z, dim=-1
        )
        actions_tp1 = torch.argmax(q_values_tp1, dim=-1, keepdim=True)
        # B x num_heads x 1

        indices_tp1 = \
            actions_tp1.unsqueeze(-1).repeat(1, 1, 1, self.num_atoms)
        # B x num_heads x 1 x num_atoms

        logits_tp1 = all_logits_tp1.gather(-2, indices_tp1).squeeze(-2)
        # B x num_heads x num_atoms
        atoms_target_t = rewards_t + (1 - done_t) * gammas * self.z

        value_loss = utils.categorical_loss(
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

        gammas = (self._gammas ** self._n_step)[None, :, None]
        # 1 x num_heads x 1

        done_t = done_t[:, None, :]  # B x 1 x 1
        rewards_t = rewards_t[:, None, :]  # B x 1 x 1
        actions_t = actions_t[:, None, None, :]  # B x 1 x 1 x 1
        indices_t = actions_t.repeat(1, self._num_heads, 1, self.num_atoms)
        # B x num_heads x 1 x num_atoms

        # critic loss (quantile regression)

        atoms_t = self.critic(states_t).gather(-2, indices_t).squeeze(-2)
        # B x num_heads x num_atoms

        all_atoms_tp1 = self.target_critic(states_tp1).detach()
        # B x num_heads x num_actions x num_atoms

        q_values_tp1 = all_atoms_tp1.mean(dim=-1)
        # B x num_heads x num_actions
        actions_tp1 = torch.argmax(q_values_tp1, dim=-1, keepdim=True)
        # B x num_heads x 1
        indices_tp1 = actions_tp1.unsqueeze(-1).repeat(1, 1, 1, self.num_atoms)
        # B x num_heads x 1 x num_atoms
        atoms_tp1 = all_atoms_tp1.gather(-2, indices_tp1).squeeze(-2)
        # B x num_heads x num_atoms
        atoms_target_t = rewards_t + (1 - done_t) * gammas * atoms_tp1

        value_loss = utils.quantile_loss(
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
