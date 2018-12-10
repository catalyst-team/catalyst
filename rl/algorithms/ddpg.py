import torch
from catalyst.rl.algorithms.base import BaseAlgorithm, \
    prepare_for_trainer as base_prepare_for_trainer, \
    prepare_for_sampler as base_prepare_for_sampler


class DDPG(BaseAlgorithm):
    def _init(self, **kwargs):
        super()._init(**kwargs)

    def train(self, batch, actor_update=True, critic_update=True):
        states_t, actions_t, rewards_t, states_tp1, done_t = \
            batch["state"], batch["action"], batch["reward"], \
            batch["next_state"], batch["done"]

        states_t = self.to_tensor(states_t)
        actions_t = self.to_tensor(actions_t)
        rewards_t = self.to_tensor(rewards_t).unsqueeze(1)
        states_tp1 = self.to_tensor(states_tp1)
        done_t = self.to_tensor(done_t).unsqueeze(1)

        # actor loss
        policy_loss = -torch.mean(self.critic(states_t, self.actor(states_t)))

        # critic loss
        q_values_tp1 = self.target_critic(
            states_tp1,
            self.target_actor(states_tp1).detach(),
        )

        gamma = self.gamma**self.n_step
        q_target_t = (rewards_t + (1 - done_t) * gamma * q_values_tp1).detach()
        q_values_t = self.critic(states_t, actions_t)
        value_loss = self.critic_criterion(q_values_t, q_target_t).mean()

        metrics = self.update_step(
            policy_loss=policy_loss,
            value_loss=value_loss,
            actor_update=actor_update,
            critic_update=critic_update
        )

        return metrics

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


def prepare_for_trainer(config, algo=DDPG):
    return base_prepare_for_trainer(config, algo=algo)


def prepare_for_sampler(config):
    return base_prepare_for_sampler(config)
