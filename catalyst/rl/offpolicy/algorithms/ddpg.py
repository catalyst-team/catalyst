import torch
from .actor_critic import OffpolicyActorCritic
from catalyst.rl import utils


class DDPG(OffpolicyActorCritic):
    """
    Swiss Army knife DDPG algorithm.
    """

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
        gammas = self._gammas ** self._n_step
        # actor loss

        # For now we have the same actor for all heads of the critic
        policy_loss = -torch.mean(self.critic(states_t, self.actor(states_t)))

        # critic loss
        q_values_t = self.critic(
            states_t, actions_t).squeeze_(dim=2)  # B x num_heads x 1
        q_values_tp1 = self.target_critic(
            states_tp1, self.target_actor(states_tp1)
        ).squeeze_(dim=2).detach()  # B x num_heads x 1
        done_t = done_t[:, None, :]  # B x 1 x 1
        rewards_t = rewards_t[:, None, :]  # B x 1 x 1
        gammas = gammas[None, :, None]   # 1 x num_heads x 1

        q_target_t = rewards_t + (1 - done_t) * gammas * q_values_tp1

        value_loss = self.critic_criterion(q_values_t, q_target_t).mean()

        return policy_loss, value_loss

    def _categorical_loss(
        self, states_t, actions_t, rewards_t, states_tp1, done_t
    ):
        gammas = self._gammas ** self._n_step

        # actor loss
        # For now we have the same actor for all heads of the critic
        logits_tp0 = self.critic(states_t, self.actor(states_t))
        probs_tp0 = torch.softmax(logits_tp0, dim=-1)
        q_values_tp0 = torch.sum(probs_tp0 * self.z, dim=-1)
        policy_loss = -torch.mean(q_values_tp0)

        # critic loss (kl-divergence between categorical distributions)

        done_t = done_t[:, None, :]
        # B x 1 x 1
        rewards_t = rewards_t[:, None, :]
        # B x 1 x 1
        gammas = gammas[None, :, None]
        # 1 x num_heads x 1

        logits_t = self.critic(states_t, actions_t).squeeze_(dim=2)
        # B x num_heads x num_atoms
        logits_tp1 = self.target_critic(
            states_tp1, self.target_actor(states_tp1)
        ).squeeze_(dim=2).detach()
        # B x num_heads x num_atoms
        atoms_target_t = rewards_t + (1 - done_t) * gammas * self.z
        # B x num_heads x num_atoms

        value_loss = utils.categorical_loss(
            logits_t.view(-1, self.num_atoms),
            logits_tp1.view(-1, self.num_atoms),
            atoms_target_t.view(-1, self.num_atoms),
            self.z,
            self.delta_z,
            self.v_min, self.v_max
        )

        return policy_loss, value_loss

    def _quantile_loss(
        self, states_t, actions_t, rewards_t, states_tp1, done_t
    ):
        gammas = self._gammas ** self._n_step

        # actor loss
        policy_loss = -torch.mean(self.critic(states_t, self.actor(states_t)))

        # critic loss (quantile regression)
        atoms_t = self.critic(states_t, actions_t).squeeze_(dim=2)
        # B x num_heads x num_atoms
        atoms_tp1 = self.target_critic(
            states_tp1, self.target_actor(states_tp1)
        ).squeeze_(dim=2).detach()
        # B x num_heads x num_atoms

        done_t = done_t[:, None, :]
        # B x 1 x 1
        rewards_t = rewards_t[:, None, :]
        # B x 1 x 1
        gammas = gammas[None, :, None]
        # 1 x num_heads x 1

        atoms_target_t = rewards_t + (1 - done_t) * gammas * atoms_tp1

        value_loss = utils.quantile_loss(
            atoms_t.view(-1, self.num_atoms),
            atoms_target_t.view(-1, self.num_atoms),
            self.tau, self.num_atoms,
            self.critic_criterion
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
