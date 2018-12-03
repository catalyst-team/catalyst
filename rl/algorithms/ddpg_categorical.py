import torch
import torch.nn.functional as F
from catalyst.rl.algorithms.base import \
    prepare_for_trainer as base_prepare_for_trainer
from catalyst.rl.algorithms.ddpg import DDPG


def ce_with_logits(logits, target):
    return torch.sum(- target * F.log_softmax(logits, -1), -1)


class CategoricalDDPG(DDPG):
    def _init(self, values_range=(-10., 10.), **kwargs):
        super()._init(**kwargs)

        num_atoms = self.critic.n_atoms
        v_min, v_max = values_range
        delta_z = (v_max - v_min) / (num_atoms - 1)
        z = torch.linspace(start=v_min, end=v_max, steps=num_atoms)

        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.delta_z = delta_z
        self.z = self.to_tensor(z)

    def train(self, batch, actor_update=True, critic_update=True):
        states, actions, rewards, next_states, done = \
            batch["state"], batch["action"], batch["reward"], \
            batch["next_state"], batch["done"]

        states = self.to_tensor(states)
        actions = self.to_tensor(actions)
        rewards = self.to_tensor(rewards).unsqueeze(1)
        next_states = self.to_tensor(next_states)
        done = self.to_tensor(done).unsqueeze(1)

        # actor loss
        q_logits_tp0 = self.critic(states, self.actor(states))
        probs_tp0 = F.softmax(q_logits_tp0, dim=-1)
        q_values_tp0 = torch.sum(probs_tp0 * self.z, dim=-1)
        policy_loss = -torch.mean(q_values_tp0)

        # critic loss
        q_logits_t = self.critic(states, actions)
        q_logits_tp1 = self.target_critic(
            next_states,
            self.target_actor(next_states).detach(),
        )
        probs_tp1 = F.softmax(q_logits_tp1, dim=-1)

        gamma = self.gamma ** self.n_step
        target_atoms = rewards + (1 - done) * gamma * self.z

        tz = torch.clamp(target_atoms, self.v_min, self.v_max)
        tz_z = tz[:, None, :] - self.z[None, :, None]
        tz_z = torch.clamp(
            (1.0 - (torch.abs(tz_z) / self.delta_z)), 0., 1.)
        target_probs = torch.einsum(
            "bij,bj->bi", (tz_z, probs_tp1)).detach()

        value_loss = ce_with_logits(q_logits_t, target_probs).mean()

        metrics = self.update_step(
            policy_loss=policy_loss,
            value_loss=value_loss,
            actor_update=actor_update,
            critic_update=critic_update)

        return metrics


def prepare_for_trainer(config):
    return base_prepare_for_trainer(config, algo=CategoricalDDPG)
