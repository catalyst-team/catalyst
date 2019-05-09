import numpy as np
import torch

from .actor import ActorAlgorithmSpec
from .utils import create_gamma_matrix


class REINFORCE(ActorAlgorithmSpec):
    def _init(
        self,
        max_episode_length: int = 1000,
        entropy_reg_coefficient: float = 0.
    ):
        self.entropy_reg_coefficient = entropy_reg_coefficient
        # matrix for estimating cummulative discounted returns
        # used in value loss
        self.gam_matrix = create_gamma_matrix(
            self.gamma, max_episode_length)

    def get_rollout_spec(self):
        return {
            "return": {"shape": (), "dtype": np.float32},
            "action_logprob": {"shape": (), "dtype": np.float32},
        }

    @torch.no_grad()
    def get_rollout(self, states, actions, rewards):
        states = self._to_tensor(states)
        actions = self._to_tensor(actions)
        rewards = np.array(rewards)
        trajectory_len = rewards.shape[0]

        _, logprobs = self.actor(states, logprob=actions)
        logprobs = logprobs.cpu().numpy().reshape(-1)

        returns = np.dot(
            self.gam_matrix[:trajectory_len, :trajectory_len],
            rewards)

        rollout = {
            "return": returns,
            "action_logprob": logprobs
        }
        return rollout

    def train(self, batch, actor_update=True, critic_update=True):
        states, actions, returns, action_logprobs = \
            batch["state"], batch["action"], batch["return"],\
            batch["action_logprob"]

        states = self._to_tensor(states)
        actions = self._to_tensor(actions)
        returns = self._to_tensor(returns)
        old_logprobs = self._to_tensor(action_logprobs)

        # actor loss
        _, logprobs = self.actor(states, logprob=actions)

        # REINFORCE objective function
        policy_loss = -torch.mean(logprobs * returns)

        entropy = -(torch.exp(logprobs) * logprobs).mean()
        entropy_loss = self.entropy_reg_coefficient * entropy
        policy_loss = policy_loss + entropy_loss

        # actor update
        actor_update_metrics = self.actor_update(policy_loss) or {}

        # metrics
        kl = 0.5 * (logprobs - old_logprobs).pow(2).mean()
        metrics = {
            "loss_actor": policy_loss.item(),
            "kl": kl.item(),
        }
        metrics = {**metrics, **actor_update_metrics}
        return metrics
