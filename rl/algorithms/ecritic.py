import copy
import torch
from catalyst.utils.factory import UtilsFactory
import catalyst.rl.networks.agents as agents
from catalyst.rl.algorithms.base import BaseAlgorithm, soft_update


class EnsembleCritic(BaseAlgorithm):
    def _init(
            self,
            critics,
            min_action, max_action,
            action_noise_std=0.2,
            action_noise_clip=0.5,
            **kwargs):
        super()._init(**kwargs)

        self.min_action = min_action
        self.max_action = max_action
        self.action_noise_std = action_noise_std
        self.action_noise_clip = action_noise_clip

        critics = [x.to(self._device) for x in critics]
        critics_optimizer = [
            UtilsFactory.create_optimizer(x, **self.critic_optimizer_params)
            for x in critics]
        critics_scheduler = [
            UtilsFactory.create_scheduler(x, **self.critic_scheduler_params)
            for x in critics_optimizer]
        target_critics = [
            copy.deepcopy(x).to(self._device)
            for x in critics]

        self.critics = [self.critic] + critics
        self.critics_optimizer = [self.critic_optimizer] + critics_optimizer
        self.critics_scheduler = [self.critic_scheduler] + critics_scheduler
        self.target_critics = [self.target_critic] + target_critics

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
        q_values_tp0 = [x(states_t, actions_tp0) for x in self.critics]
        q_values_tp0_min = torch.cat(q_values_tp0, dim=-1).min(dim=-1)[0]
        policy_loss = -torch.mean(q_values_tp0_min)

        # critic loss
        actions_tp1 = self.target_actor(states_tp1).detach()
        action_noise = torch.normal(
            mean=torch.zeros_like(actions_tp1), std=self.action_noise_std)
        action_noise = action_noise.clamp(
            -self.action_noise_clip, self.action_noise_clip)
        actions_tp1 = actions_tp1 + action_noise
        actions_tp1 = actions_tp1.clamp(
            self.min_action, self.max_action)

        q_values_tp1 = torch.cat([
            x(states_tp1, actions_tp1)
            for x in self.target_critics],
            dim=-1)
        q_values_tp1 = q_values_tp1.min(dim=1, keepdim=True)[0]

        gamma = self.gamma ** self.n_step
        q_target_t = (rewards_t + (1 - done_t) * gamma * q_values_tp1).detach()
        q_values_t = [x(states_t, actions_t) for x in self.critics]
        value_loss = [
            self.critic_criterion(x, q_target_t).mean()
            for x in q_values_t]

        metrics = self.update_step(
            policy_loss=policy_loss,
            value_loss=value_loss,
            actor_update=actor_update,
            critic_update=critic_update)

        return metrics

    def update_step(
            self, policy_loss, value_loss,
            actor_update=True, critic_update=True):
        # actor update
        actor_update_metrics = {}
        if actor_update:
            actor_update_metrics = self.actor_update(policy_loss) or {}

        # critic update
        critic_update_metrics = {}
        if critic_update:
            critic_update_metrics = self.critic_update(value_loss) or {}

        loss = policy_loss
        for l_ in value_loss:
            loss += l_

        metrics = {
            f"loss_critic{i}": x.item()
            for i, x in enumerate(value_loss)}
        metrics = {
            **{
                "loss": loss.item(),
                "loss_actor": policy_loss.item()
            }, **metrics}
        metrics = {**metrics, **actor_update_metrics, **critic_update_metrics}

        return metrics

    def target_critic_update(self):
        for target, source in zip(self.target_critics, self.critics):
            soft_update(target, source, self.critic_tau)

    def critic_update(self, loss):
        metrics = {}
        for i in range(len(self.critics)):
            self.critics[i].zero_grad()
            self.critics_optimizer[i].zero_grad()
            loss[i].backward()
            if self.critic_grad_clip is not None:
                self.critic_grad_clip(self.critics[i].parameters())
            self.critics_optimizer[i].step()
            if self.critics_scheduler[i] is not None:
                self.critics_scheduler[i].step()
                metrics[f"lr_critic{i}"] = self.critics_scheduler[i].get_lr()[0]
        return metrics

    def load_checkpoint(self, filepath, load_optimizer=True):
        super().load_checkpoint(filepath, load_optimizer)

        checkpoint = UtilsFactory.load_checkpoint(filepath)
        key = "critics"
        for i in range(len(self.critics)):
            value_l = getattr(self, key, None)
            value_l = value_l[i] if value_l is not None else None
            if value_l is not None:
                value_r = checkpoint[f"{key}{i}_state_dict"]
                value_l.load_state_dict(value_r)
            if load_optimizer:
                for key2 in ["optimizer", "scheduler"]:
                    key2 = f"{key}_{key2}"
                    value_l = getattr(self, key2, None)
                    if value_l is not None:
                        value_r = checkpoint[f"{key2}_state_dict"]
                        value_l.load_state_dict(value_r)

    def prepare_checkpoint(self):
        checkpoint = {}

        for key in ["actor", "critic"]:
            checkpoint[f"{key}_state_dict"] = getattr(self, key).state_dict()
            for key2 in ["optimizer", "scheduler"]:
                key2 = f"{key}_{key2}"
                value2 = getattr(self, key2, None)
                if value2 is not None:
                    checkpoint[f"{key2}_state_dict"] = value2.state_dict()

        key = "critics"
        for i in range(len(self.critics)):
            value = getattr(self, key)
            checkpoint[f"{key}{i}_state_dict"] = value[i].state_dict()
            for key2 in ["optimizer", "scheduler"]:
                key2 = f"{key}_{key2}"
                value2 = getattr(self, key2, None)
                if value2 is not None:
                    checkpoint[f"{key2}{i}_state_dict"] = value2[i].state_dict()

        return checkpoint


def prepare_for_trainer(config, algo=EnsembleCritic):
    config_ = config.copy()

    actor_state_shape = (
        config_["shared"]["history_len"],
        config_["shared"]["state_size"],)
    actor_action_size = config_["shared"]["action_size"]
    n_step = config_["shared"]["n_step"]
    gamma = config_["shared"]["gamma"]
    history_len = config_["shared"]["history_len"]
    trainer_state_shape = (config_["shared"]["state_size"],)
    trainer_action_shape = (config_["shared"]["action_size"],)

    actor_fn = config_["actor"].pop("actor", None)
    actor_fn = getattr(agents, actor_fn)
    actor = actor_fn(
        state_shape=actor_state_shape,
        action_size=actor_action_size,
        **config_["actor"])

    critic_fn = config_["critic"].pop("critic", None)
    critic_fn = getattr(agents, critic_fn)
    critic = critic_fn(
        state_shape=actor_state_shape,
        action_size=actor_action_size,
        **config_["critic"])

    n_critics = config_["algorithm"].pop("n_critics", 2)
    critics = [
        critic_fn(
            state_shape=actor_state_shape,
            action_size=actor_action_size,
            **config_["critic"])
        for _ in range(n_critics-1)]

    algorithm = algo(
        **config_["algorithm"],
        actor=actor, critic=critic, critics=critics,
        n_step=n_step, gamma=gamma)

    kwargs = {
        "algorithm": algorithm,
        "state_shape": trainer_state_shape,
        "action_shape": trainer_action_shape,
        "n_step": n_step,
        "gamma": gamma,
        "history_len": history_len
    }

    return kwargs


def prepare_for_sampler(config):
    config_ = config.copy()

    actor_state_shape = (
        config_["shared"]["history_len"],
        config_["shared"]["state_size"],)
    actor_action_size = config_["shared"]["action_size"]

    actor_fn = config_["actor"].pop("actor", None)
    actor_fn = getattr(agents, actor_fn)
    actor = actor_fn(
        **config_["actor"],
        state_shape=actor_state_shape,
        action_size=actor_action_size)

    history_len = config_["shared"]["history_len"]

    kwargs = {
        "actor": actor,
        "history_len": history_len
    }

    return kwargs
