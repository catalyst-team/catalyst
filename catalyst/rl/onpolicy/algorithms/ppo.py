import numpy as np
import torch


from .actor_critic import OnpolicyActorCritic
from catalyst.rl import utils


class PPO(OnpolicyActorCritic):

    def _init(
        self,
        use_value_clipping: bool = True,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        entropy_reg_coefficient: float = 0.
    ):
        self.use_value_clipping = use_value_clipping
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        critic_distribution = self.critic.distribution
        self._value_loss_fn = self._base_value_loss
        self._num_atoms = self.critic.num_atoms
        self._num_heads = self.critic.num_heads
        self._hyperbolic_constant = self.critic.hyperbolic_constant
        self._gammas = \
            utils.hyperbolic_gammas(
                self._gamma,
                self._hyperbolic_constant,
                self._num_heads
            )
        self.entropy_reg_coefficient = entropy_reg_coefficient

        if critic_distribution == "categorical":
            values_range = self.critic.values_range
            self.v_min, self.v_max = values_range
            self.delta_z = (self.v_max - self.v_min) / (self._num_atoms - 1)
            z = torch.linspace(
                start=self.v_min, end=self.v_max, steps=self._num_atoms
            )
            self.z = utils.any2device(z, device=self._device)
            self._value_loss_fn = self._categorical_value_loss
        elif critic_distribution == "quantile":
            tau_min = 1 / (2 * self._num_atoms)
            tau_max = 1 - tau_min
            tau = torch.linspace(
                start=tau_min, end=tau_max, steps=self._num_atoms
            )
            self.tau = utils.any2device(tau, device=self._device)
            self._value_loss_fn = self._quantile_value_loss

        if not self.use_value_clipping:
            assert self.critic_criterion is not None

    def _value_loss(self, values, returns, old_values):
        if self.use_value_clipping:
            values_clip = old_values + torch.clamp(
                values - old_values, -self.clip_eps, self.clip_eps)
            value_loss_unclipped = (values - returns).pow(2)
            value_loss_clipped = (values_clip - returns).pow(2)
            value_loss = 0.5 * torch.max(
                value_loss_unclipped, value_loss_clipped).mean()
        else:
            value_loss = self.critic_criterion(
                values[:, None], returns[:, None]).mean()

        return value_loss

    def _base_value_loss(self, states, returns, old_values):
        values = self.critic(states).squeeze_(dim=2)
        value_loss = self._value_loss(
            values.squeeze_(-1),
            returns,
            old_values.squeeze_(-1))
        return value_loss

    def _categorical_value_loss(self, states, returns, old_logits):
        # @TODO: WIP, no guaranties
        logits = self.critic(states).squeeze_(dim=2)
        probs = torch.softmax(logits, dim=-1)
        values = torch.sum(probs * self.z, dim=-1)

        old_probs = torch.softmax(old_logits, dim=-1)
        old_values = torch.sum(old_probs * self.z, dim=-1)

        value_loss = self._value_loss(values, returns, old_values)
        return value_loss

    def _quantile_value_loss(self, states, returns, old_atoms):
        # @TODO: WIP, no guaranties
        # how to propagate atoms loss correctly?
        atoms = self.critic(states).squeeze_(dim=2)
        values = torch.mean(atoms, dim=-1)

        old_values = torch.mean(old_atoms, dim=-1)

        value_loss = self._value_loss(values, returns, old_values)
        return value_loss

    def get_rollout_spec(self):
        return {
            "return": {
                "shape": (self._num_heads, ),
                "dtype": np.float32
            },
            "value": {
                "shape": (self._num_heads, self._num_atoms),
                "dtype": np.float32
            },
            "advantage": {
                "shape": (self._num_heads, self._num_atoms),
                "dtype": np.float32
            },
            "action_logprob": {"shape": (), "dtype": np.float32},
        }

    @torch.no_grad()
    def get_rollout(self, states, actions, rewards, dones):
        trajectory_len = \
            rewards.shape[0] if dones[-1] else rewards.shape[0] - 1
        states_len = states.shape[0]

        states = utils.any2device(states, device=self._device)
        actions = utils.any2device(actions, device=self._device)
        rewards = np.array(rewards)[:trajectory_len]
        values = torch.zeros(
            (states_len + 1, self._num_heads, self._num_atoms)).\
            to(self._device)
        values[:states_len, :] = self.critic(states).squeeze_(dim=2)
        # Each column corresponds to a different gamma
        values = values.cpu().numpy()[:trajectory_len+1, ...]
        _, logprobs = self.actor(states, logprob=actions)
        logprobs = logprobs.cpu().numpy().reshape(-1)[:trajectory_len]
        # deltas = rewards[:, None] + self._gammas * values[1:] - values[:-1]
        deltas = rewards[:, None, None] \
            + self._gammas[:, None] * values[1:] - values[:-1]
        # len x num_heads

        # For each gamma in the list of gammas compute the
        # advantage and returns
        # len x num_heads x num_atoms
        advantages = np.stack([
            utils.geometric_cumsum(gamma, deltas[:, i])
            for i, gamma in enumerate(self._gammas)
        ], axis=1)
        # len x num_heads x 1
        returns = np.stack([
            utils.geometric_cumsum(gamma * self.gae_lambda, rewards)[0]
            for gamma in self._gammas
        ], axis=1)

        # final rollout
        rollout = {
            "return": returns,
            "value": values[:trajectory_len],
            "advantage": advantages,
            "action_logprob": logprobs
        }

        return rollout

    def postprocess_buffer(self, buffers, len):
        adv_centered = \
            buffers["advantage"][:len] \
            - buffers["advantage"][:len].mean(axis=0)
        adv_std = buffers["advantage"][:len].std(axis=0)
        buffers["advantage"][:len] = adv_centered / (adv_std + 1e-6)

    def train(self, batch, **kwargs):
        states, actions, returns, values, advantages, action_logprobs = \
            batch["state"], batch["action"], batch["return"], \
            batch["value"], batch["advantage"], batch["action_logprob"]

        states = utils.any2device(states, device=self._device)
        actions = utils.any2device(actions, device=self._device)
        returns = utils.any2device(returns, device=self._device)
        old_values = utils.any2device(values, device=self._device)
        advantages = utils.any2device(advantages, device=self._device)
        old_logprobs = utils.any2device(action_logprobs, device=self._device)

        # critic loss
        value_loss = self._value_loss_fn(states, returns, old_values)

        # actor loss
        _, logprobs = self.actor(states, logprob=actions)

        ratio = torch.exp(logprobs - old_logprobs)
        # The same ratio for each head of the critic
        policy_loss_unclipped = advantages * ratio[:, None, None]
        policy_loss_clipped = advantages * torch.clamp(
            ratio[:, None, None], 1.0 - self.clip_eps, 1.0 + self.clip_eps)
        policy_loss = -torch.min(
            policy_loss_unclipped, policy_loss_clipped).mean()

        entropy = -(torch.exp(logprobs) * logprobs).mean()
        entropy_loss = self.entropy_reg_coefficient * entropy
        policy_loss = policy_loss + entropy_loss

        # actor update
        actor_update_metrics = self.actor_update(policy_loss) or {}

        # critic update
        critic_update_metrics = self.critic_update(value_loss) or {}

        # metrics
        kl = 0.5 * (logprobs - old_logprobs).pow(2).mean()
        clipped_fraction = \
            (torch.abs(ratio - 1.0) > self.clip_eps).float().mean()
        metrics = {
            "loss_actor": policy_loss.item(),
            "loss_critic": value_loss.item(),
            "kl": kl.item(),
            "clipped_fraction": clipped_fraction.item()
        }
        metrics = {**metrics, **actor_update_metrics, **critic_update_metrics}
        return metrics
