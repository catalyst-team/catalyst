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
        entropy_regularization: float = None
    ):
        self.use_value_clipping = use_value_clipping
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.entropy_regularization = entropy_regularization

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
        # 1 x num_heads x 1
        self._gammas_torch = utils.any2device(
            self._gammas, device=self._device
        )[None, :, None]

        if critic_distribution == "categorical":
            self.num_atoms = self.critic.num_atoms
            values_range = self.critic.values_range
            self.v_min, self.v_max = values_range
            self.delta_z = (self.v_max - self.v_min) / (self._num_atoms - 1)
            z = torch.linspace(
                start=self.v_min, end=self.v_max, steps=self._num_atoms
            )
            self.z = utils.any2device(z, device=self._device)
            self._value_loss_fn = self._categorical_value_loss
        elif critic_distribution == "quantile":
            assert self.critic_criterion is not None

            self.num_atoms = self.critic.num_atoms
            tau_min = 1 / (2 * self._num_atoms)
            tau_max = 1 - tau_min
            tau = torch.linspace(
                start=tau_min, end=tau_max, steps=self._num_atoms
            )
            self.tau = utils.any2device(tau, device=self._device)
            self._value_loss_fn = self._quantile_value_loss

        if not self.use_value_clipping:
            assert self.critic_criterion is not None

    def _value_loss(self, values_tp0, values_t, returns_t):
        if self.use_value_clipping:
            values_clip = values_t + torch.clamp(
                values_tp0 - values_t, -self.clip_eps, self.clip_eps
            )
            value_loss_unclipped = (values_tp0 - returns_t).pow(2)
            value_loss_clipped = (values_clip - returns_t).pow(2)
            value_loss = 0.5 * torch.max(
                value_loss_unclipped, value_loss_clipped
            ).mean()
        else:
            value_loss = self.critic_criterion(values_tp0, returns_t).mean()

        return value_loss

    def _base_value_loss(
        self, states_t, values_t, returns_t, states_tp1, done_t
    ):
        values_tp0 = self.critic(states_t).squeeze_(dim=2)
        value_loss = self._value_loss(values_tp0, values_t, returns_t)
        return value_loss

    def _categorical_value_loss(
        self, states_t, logits_t, returns_t, states_tp1, done_t
    ):
        # @TODO: WIP, no guaranties
        logits_tp0 = self.critic(states_t).squeeze_(dim=2)
        probs_tp0 = torch.softmax(logits_tp0, dim=-1)
        values_tp0 = torch.sum(probs_tp0 * self.z, dim=-1, keepdim=True)

        probs_t = torch.softmax(logits_t, dim=-1)
        values_t = torch.sum(probs_t * self.z, dim=-1, keepdim=True)

        value_loss = 0.5 * self._value_loss(values_tp0, values_t, returns_t)

        # B x num_heads x num_atoms
        logits_tp1 = self.critic(states_tp1).squeeze_(dim=2).detach()
        # B x num_heads x num_atoms
        atoms_target_t = returns_t + (1 - done_t) * self._gammas_torch * self.z

        value_loss += 0.5 * utils.categorical_loss(
            logits_tp0.view(-1, self.num_atoms),
            logits_tp1.view(-1, self.num_atoms),
            atoms_target_t.view(-1, self.num_atoms), self.z, self.delta_z,
            self.v_min, self.v_max
        )

        return value_loss

    def _quantile_value_loss(
        self, states_t, atoms_t, returns_t, states_tp1, done_t
    ):
        # @TODO: WIP, no guaranties
        atoms_tp0 = self.critic(states_t).squeeze_(dim=2)
        values_tp0 = torch.mean(atoms_tp0, dim=-1, keepdim=True)

        values_t = torch.mean(atoms_t, dim=-1, keepdim=True)

        value_loss = 0.5 * self._value_loss(values_tp0, values_t, returns_t)

        # B x num_heads x num_atoms
        atoms_tp1 = self.critic(states_tp1).squeeze_(dim=2).detach()
        # B x num_heads x num_atoms
        atoms_target_t = returns_t \
            + (1 - done_t) * self._gammas_torch * atoms_tp1

        value_loss += 0.5 * utils.quantile_loss(
            atoms_tp0.view(-1, self.num_atoms),
            atoms_target_t.view(-1, self.num_atoms), self.tau, self.num_atoms,
            self.critic_criterion
        )

        return value_loss

    def get_rollout_spec(self):
        return {
            "action_logprob": {
                "shape": (),
                "dtype": np.float32
            },
            "advantage": {
                "shape": (self._num_heads, self._num_atoms),
                "dtype": np.float32
            },
            "done": {
                "shape": (),
                "dtype": np.bool
            },
            "return": {
                "shape": (self._num_heads, ),
                "dtype": np.float32
            },
            "value": {
                "shape": (self._num_heads, self._num_atoms),
                "dtype": np.float32
            },
        }

    @torch.no_grad()
    def get_rollout(self, states, actions, rewards, dones):
        assert len(states) == len(actions) == len(rewards) == len(dones)

        trajectory_len = \
            rewards.shape[0] if dones[-1] else rewards.shape[0] - 1
        states_len = states.shape[0]

        states = utils.any2device(states, device=self._device)
        actions = utils.any2device(actions, device=self._device)
        rewards = np.array(rewards)[:trajectory_len]
        values = torch.zeros(
            (states_len + 1, self._num_heads, self._num_atoms)).\
            to(self._device)
        values[:states_len, ...] = self.critic(states).squeeze_(dim=2)
        # Each column corresponds to a different gamma
        values = values.cpu().numpy()[:trajectory_len + 1, ...]
        _, logprobs = self.actor(states, logprob=actions)
        logprobs = logprobs.cpu().numpy().reshape(-1)[:trajectory_len]
        # len x num_heads
        deltas = rewards[:, None, None] \
            + self._gammas[:, None] * values[1:] - values[:-1]

        # For each gamma in the list of gammas compute the
        # advantage and returns
        # len x num_heads x num_atoms
        advantages = np.stack(
            [
                utils.geometric_cumsum(gamma * self.gae_lambda, deltas[:, i])
                for i, gamma in enumerate(self._gammas)
            ],
            axis=1
        )

        # len x num_heads
        returns = np.stack(
            [
                utils.geometric_cumsum(gamma, rewards[:, None])[:, 0]
                for gamma in self._gammas
            ],
            axis=1
        )

        # final rollout
        dones = dones[:trajectory_len]
        values = values[:trajectory_len]
        assert len(logprobs) == len(advantages) \
            == len(dones) == len(returns) == len(values)
        rollout = {
            "action_logprob": logprobs,
            "advantage": advantages,
            "done": dones,
            "return": returns,
            "value": values,
        }

        return rollout

    def postprocess_buffer(self, buffers, len):
        adv = buffers["advantage"][:len]
        adv = (adv - adv.mean(axis=0)) / (adv.std(axis=0) + 1e-8)
        buffers["advantage"][:len] = adv

    def train(self, batch, **kwargs):
        (
            states_t, actions_t, returns_t, states_tp1, done_t, values_t,
            advantages_t, action_logprobs_t
        ) = (
            batch["state"], batch["action"], batch["return"],
            batch["state_tp1"], batch["done"], batch["value"],
            batch["advantage"], batch["action_logprob"]
        )

        states_t = utils.any2device(states_t, device=self._device)
        actions_t = utils.any2device(actions_t, device=self._device)
        returns_t = utils.any2device(
            returns_t, device=self._device
        ).unsqueeze_(-1)
        states_tp1 = utils.any2device(states_tp1, device=self._device)
        done_t = utils.any2device(done_t, device=self._device)[:, None, None]

        values_t = utils.any2device(values_t, device=self._device)
        advantages_t = utils.any2device(advantages_t, device=self._device)
        action_logprobs_t = utils.any2device(
            action_logprobs_t, device=self._device
        )

        # critic loss
        value_loss = self._value_loss_fn(
            states_t, values_t, returns_t, states_tp1, done_t
        )

        # actor loss
        _, action_logprobs_tp0 = self.actor(states_t, logprob=actions_t)

        ratio = torch.exp(action_logprobs_tp0 - action_logprobs_t)
        ratio = ratio[:, None, None]
        # The same ratio for each head of the critic
        policy_loss_unclipped = advantages_t * ratio
        policy_loss_clipped = advantages_t * torch.clamp(
            ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps
        )
        policy_loss = -torch.min(
            policy_loss_unclipped, policy_loss_clipped).mean()

        if self.entropy_regularization is not None:
            entropy = -(
                torch.exp(action_logprobs_tp0) * action_logprobs_tp0).mean()
            entropy_loss = self.entropy_regularization * entropy
            policy_loss = policy_loss + entropy_loss

        # actor update
        actor_update_metrics = self.actor_update(policy_loss) or {}

        # critic update
        critic_update_metrics = self.critic_update(value_loss) or {}

        # metrics
        kl = 0.5 * (action_logprobs_tp0 - action_logprobs_t).pow(2).mean()
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
