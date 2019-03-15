import torch
import torch.nn.functional as F
from catalyst.rl.offpolicy.algorithms.core_dqn import Algorithm
from catalyst.rl.offpolicy.algorithms.utils import categorical_loss, \
    quantile_loss


class DQN(Algorithm):

    def _init(self, **kwargs):
        super()._init(**kwargs)
        self._loss_fn = self._base_loss

    def _base_loss(self, states_t, actions_t, rewards_t, states_tp1, done_t):
        gamma = self.gamma**self.n_step

        # critic loss
        q_values_t = self.critic(states_t).gather(-1, actions_t)
        q_values_tp1 = self.target_critic(states_tp1).max(-1, keepdim=True)[0]
        q_target_t = rewards_t + (1 - done_t) * gamma * q_values_tp1.detach()
        value_loss = self.critic_criterion(q_values_t, q_target_t).mean()

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
