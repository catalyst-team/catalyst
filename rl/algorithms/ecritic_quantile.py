import torch
from catalyst.rl.algorithms.utils import quantile_loss
from catalyst.rl.algorithms.ecritic import EnsembleCritic, \
    prepare_for_trainer as base_prepare_for_trainer, \
    prepare_for_sampler as base_prepare_for_sampler


class QuantileEnsembleCritic(EnsembleCritic):
    def _init(self, **kwargs):
        super()._init(**kwargs)

        self.num_atoms = self.critic.n_atoms
        tau_min = 1 / (2 * self.num_atoms)
        tau_max = 1 - tau_min
        tau = torch.linspace(start=tau_min, end=tau_max, steps=self.num_atoms)
        self.tau = self.to_tensor(tau)

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
        actions_tp0 = self.actor(states_t)
        atoms_tp0 = [
            x(states_t, actions_tp0).unsqueeze_(-1)
            for x in self.critics]
        atoms_min_tp0 = torch.cat(atoms_tp0, dim=-1).mean(dim=1).min(dim=1)[0]
        policy_loss = -torch.mean(atoms_min_tp0)

        # critic loss
        actions_tp1 = self.target_actor(states_tp1).detach()
        action_noise = torch.normal(
            mean=torch.zeros_like(actions_tp1), std=self.action_noise_std)
        action_noise = action_noise.clamp(
            -self.action_noise_clip, self.action_noise_clip)
        actions_tp1 = actions_tp1 + action_noise
        actions_tp1 = actions_tp1.clamp(self.min_action, self.max_action)

        atoms_tp1 = torch.cat([
            x(states_tp1, actions_tp1).unsqueeze_(-1)
            for x in self.target_critics],
            dim=-1)
        atoms_min_ids_tp1 = atoms_tp1.mean(dim=1).argmin(dim=1)
        atoms_tp1 = atoms_tp1[range(len(atoms_tp1)), :, atoms_min_ids_tp1]

        gamma = self.gamma ** self.n_step
        atoms_target_t = (
                rewards_t + (1 - done_t) * gamma * atoms_tp1).detach()

        atoms_t = [x(states_t, actions_t) for x in self.critics]
        value_loss = [
            quantile_loss(
                x, atoms_target_t,
                tau=self.tau, n_atoms=self.num_atoms,
                criterion=self.critic_criterion).mean()
            for x in atoms_t]

        metrics = self.update_step(
            policy_loss=policy_loss,
            value_loss=value_loss,
            actor_update=actor_update,
            critic_update=critic_update)

        return metrics


def prepare_for_trainer(config):
    return base_prepare_for_trainer(config, algo=QuantileEnsembleCritic)


def prepare_for_sampler(config):
    return base_prepare_for_sampler(config)
