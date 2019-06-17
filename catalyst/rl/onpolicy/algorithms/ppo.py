import numpy as np
import torch


from .actor_critic import OnpolicyActorCritic
from catalyst.rl.utils import hyperbolic_gammas, geometric_cumsum


class PPO(OnpolicyActorCritic):

    def _init(
        self,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        entropy_reg_coefficient: float = 0.
    ):
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self._num_heads = self.critic.num_heads
        self._hyperbolic_constant = self.critic.hyperbolic_constant
        self._gammas = \
            hyperbolic_gammas(
                self._gamma,
                self._hyperbolic_constant,
                self._num_heads
            )
        self.entropy_reg_coefficient = entropy_reg_coefficient

    def get_rollout_spec(self):
        return {
            "return": {"shape": (self._num_heads,), "dtype": np.float32},
            "value": {"shape": (self._num_heads,), "dtype": np.float32},
            "advantage": {"shape": (self._num_heads,), "dtype": np.float32},
            "action_logprob": {"shape": (), "dtype": np.float32},
        }

    @torch.no_grad()
    def get_rollout(self, states, actions, rewards, dones):
        trajectory_len = \
            rewards.shape[0] if dones[-1] else rewards.shape[0] - 1

        states = self._to_tensor(states)
        actions = self._to_tensor(actions)
        rewards = np.array(rewards)[:trajectory_len]

        values = torch.zeros((states.shape[0] + 1, self._num_heads)).\
            to(self._device)
        values[:states.shape[0], :] = self.critic(states).squeeze(-1)
        # Each column corresponds to a different gamma
        values = values.cpu().numpy()[:trajectory_len+1, :]
        _, logprobs = self.actor(states, logprob=actions)
        logprobs = logprobs.cpu().numpy().reshape(-1)[:trajectory_len]
        deltas = rewards[:, None] + self._gammas * values[1:] - values[:-1]
        # len x num_heads

        # For each gamma in the list of gammas compute the
        # advantage and returns
        advantages = np.stack([
            geometric_cumsum(gamma, deltas[:, i])[0]
            for i, gamma in enumerate(self._gammas)
        ], axis=1)  # len x num_heads
        returns = np.stack([
            geometric_cumsum(gamma * self.gae_lambda, rewards)[0]
            for gamma in self._gammas
        ], axis=1)  # len x num_heads

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

        states = self._to_tensor(states)
        actions = self._to_tensor(actions)
        returns = self._to_tensor(returns)
        old_values = self._to_tensor(values)
        advantages = self._to_tensor(advantages)
        old_logprobs = self._to_tensor(action_logprobs)

        # critic loss
        values = self.critic(states).squeeze(-1)

        values_clip = old_values + torch.clamp(
            values - old_values, -self.clip_eps, self.clip_eps)
        value_loss_unclipped = (values - returns).pow(2)
        value_loss_clipped = (values_clip - returns).pow(2)
        value_loss = 0.5 * torch.max(
            value_loss_unclipped, value_loss_clipped).mean()

        # actor loss
        _, logprobs = self.actor(states, logprob=actions)

        ratio = torch.exp(logprobs - old_logprobs)
        # The same ratio for each head of the critic
        policy_loss_unclipped = advantages * ratio[:, None]
        policy_loss_clipped = advantages * torch.clamp(
            ratio[:, None], 1.0 - self.clip_eps, 1.0 + self.clip_eps)
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
