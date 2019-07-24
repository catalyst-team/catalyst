import numpy as np
import torch

from .actor import OnpolicyActor
from catalyst.rl import utils


class REINFORCE(OnpolicyActor):
    def _init(self, entropy_regularization: float = None):
        self.entropy_regularization = entropy_regularization

    def get_rollout_spec(self):
        return {
            "return": {
                "shape": (),
                "dtype": np.float32
            },
            "action_logprob": {
                "shape": (),
                "dtype": np.float32
            },
        }

    @torch.no_grad()
    def get_rollout(self, states, actions, rewards, dones):
        assert len(states) == len(actions) == len(rewards) == len(dones)

        trajectory_len = \
            rewards.shape[0] if dones[-1] else rewards.shape[0] - 1

        states = utils.any2device(states, device=self._device)
        actions = utils.any2device(actions, device=self._device)
        rewards = np.array(rewards)[:trajectory_len]

        _, logprobs = self.actor(states, logprob=actions)
        logprobs = logprobs.cpu().numpy().reshape(-1)[:trajectory_len]

        returns = utils.geometric_cumsum(self.gamma, rewards[:, None])[:, 0]

        assert len(returns) == len(logprobs)
        rollout = {"return": returns, "action_logprob": logprobs}
        return rollout

    def postprocess_buffer(self, buffers, len):
        pass

    def train(self, batch, **kwargs):
        states, actions, returns, action_logprobs = \
            batch["state"], batch["action"], batch["return"],\
            batch["action_logprob"]

        states = utils.any2device(states, device=self._device)
        actions = utils.any2device(actions, device=self._device)
        returns = utils.any2device(returns, device=self._device)
        old_logprobs = utils.any2device(action_logprobs, device=self._device)

        # actor loss
        _, logprobs = self.actor(states, logprob=actions)

        # REINFORCE objective function
        policy_loss = -torch.mean(logprobs * returns)

        if self.entropy_regularization is not None:
            entropy = -(torch.exp(logprobs) * logprobs).mean()
            entropy_loss = self.entropy_regularization * entropy
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
