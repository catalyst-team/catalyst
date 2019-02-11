import copy
import torch
import torch.nn.functional as F
from catalyst.dl.utils import UtilsFactory
from catalyst.rl.offpolicy.algorithms.core import Algorithm
from catalyst.rl.offpolicy.algorithms.utils import categorical_loss, \
    quantile_loss, soft_update


class TD3(Algorithm):
    """
    Swiss Army knife TD3 algorithm.
    """

    def _init(
        self,
        critics,
        action_noise_std=0.2,
        action_noise_clip=0.5,
        values_range=(-10., 10.),
        critic_distribution=None,
        **kwargs
    ):
        super()._init(**kwargs)
        # hack to prevent cycle dependencies
        from catalyst.contrib.registry import Registry

        self.n_atoms = self.critic.out_features
        self._loss_fn = self._base_loss

        self.action_noise_std = action_noise_std
        self.action_noise_clip = action_noise_clip

        critics = [x.to(self._device) for x in critics]
        critics_optimizer = [
            Registry.get_optimizer(x, **self.critic_optimizer_params)
            for x in critics
        ]
        critics_scheduler = [
            Registry.get_scheduler(x, **self.critic_scheduler_params)
            for x in critics_optimizer
        ]
        target_critics = [copy.deepcopy(x).to(self._device) for x in critics]

        self.critics = [self.critic] + critics
        self.critics_optimizer = [self.critic_optimizer] + critics_optimizer
        self.critics_scheduler = [self.critic_scheduler] + critics_scheduler
        self.target_critics = [self.target_critic] + target_critics

        if critic_distribution == "quantile":
            tau_min = 1 / (2 * self.n_atoms)
            tau_max = 1 - tau_min
            tau = torch.linspace(
                start=tau_min, end=tau_max, steps=self.n_atoms
            )
            self.tau = self._to_tensor(tau)
            self._loss_fn = self._quantile_loss
        elif critic_distribution == "categorical":
            self.v_min, self.v_max = values_range
            self.delta_z = (self.v_max - self.v_min) / (self.n_atoms - 1)
            z = torch.linspace(
                start=self.v_min, end=self.v_max, steps=self.n_atoms
            )
            self.z = self._to_tensor(z)
            self._loss_fn = self._categorical_loss

    def _base_loss(self, states_t, actions_t, rewards_t, states_tp1, done_t):

        # actor loss
        actions_tp0 = self.actor(states_t)
        q_values_tp0 = [x(states_t, actions_tp0) for x in self.critics]
        q_values_tp0_min = torch.cat(q_values_tp0, dim=-1).min(dim=-1)[0]
        policy_loss = -torch.mean(q_values_tp0_min)

        # critic loss
        actions_tp1 = self.target_actor(states_tp1).detach()
        actions_tp1 = self._add_noise_to_actions(actions_tp1)
        q_values_t = [x(states_t, actions_t) for x in self.critics]
        q_values_tp1 = torch.cat(
            [x(states_tp1, actions_tp1) for x in self.target_critics], dim=-1
        )
        q_values_tp1 = q_values_tp1.min(dim=1, keepdim=True)[0].detach()
        gamma = self.gamma**self.n_step
        q_target_t = rewards_t + (1 - done_t) * gamma * q_values_tp1
        value_loss = [
            self.critic_criterion(x, q_target_t).mean() for x in q_values_t
        ]

        return policy_loss, value_loss

    def _categorical_loss(
        self, states_t, actions_t, rewards_t, states_tp1, done_t
    ):

        # actor loss
        actions_tp0 = self.actor(states_t)
        logits_tp0 = [x(states_t, actions_tp0) for x in self.critics]
        probs_tp0 = [F.softmax(x, dim=-1) for x in logits_tp0]
        q_values_tp0 = [
            torch.sum(x * self.z, dim=-1).unsqueeze_(-1) for x in probs_tp0
        ]
        q_values_tp0_min = torch.cat(q_values_tp0, dim=-1).min(dim=-1)[0]
        policy_loss = -torch.mean(q_values_tp0_min)

        # critic loss (kl-divergence between categorical distributions)
        actions_tp1 = self.target_actor(states_tp1).detach()
        actions_tp1 = self._add_noise_to_actions(actions_tp1)
        logits_t = [x(states_t, actions_t) for x in self.critics]
        logits_tp1 = [x(states_tp1, actions_tp1) for x in self.target_critics]
        probs_tp1 = [F.softmax(x, dim=-1) for x in logits_tp1]
        q_values_tp1 = [
            torch.sum(x * self.z, dim=-1).unsqueeze_(-1) for x in probs_tp1
        ]
        probs_ids_tp1_min = torch.cat(q_values_tp1, dim=-1).argmin(dim=1)

        logits_tp1 = torch.cat([x.unsqueeze(-1) for x in logits_tp1], dim=-1)
        logits_tp1 = logits_tp1[range(len(logits_tp1)), :, probs_ids_tp1_min
                                ].detach()
        gamma = self.gamma**self.n_step
        atoms_target_t = rewards_t + (1 - done_t) * gamma * self.z
        value_loss = [
            categorical_loss(
                x, logits_tp1, atoms_target_t, self.z, self.delta_z,
                self.v_min, self.v_max
            ) for x in logits_t
        ]

        return policy_loss, value_loss

    def _quantile_loss(
        self, states_t, actions_t, rewards_t, states_tp1, done_t
    ):

        # actor loss
        actions_tp0 = self.actor(states_t)
        atoms_tp0 = [
            x(states_t, actions_tp0).unsqueeze_(-1) for x in self.critics
        ]
        q_values_tp0_min = torch.cat(
            atoms_tp0, dim=-1
        ).mean(dim=1).min(dim=1)[0]
        policy_loss = -torch.mean(q_values_tp0_min)

        # critic loss (quantile regression)
        actions_tp1 = self.target_actor(states_tp1).detach()
        actions_tp1 = self._add_noise_to_actions(actions_tp1)
        atoms_t = [x(states_t, actions_t) for x in self.critics]
        atoms_tp1 = torch.cat(
            [
                x(states_tp1, actions_tp1).unsqueeze_(-1)
                for x in self.target_critics
            ],
            dim=-1
        )
        atoms_ids_tp1_min = atoms_tp1.mean(dim=1).argmin(dim=1)
        atoms_tp1 = atoms_tp1[range(len(atoms_tp1)), :, atoms_ids_tp1_min
                              ].detach()
        gamma = self.gamma**self.n_step
        atoms_target_t = rewards_t + (1 - done_t) * gamma * atoms_tp1
        value_loss = [
            quantile_loss(
                x, atoms_target_t, self.tau, self.n_atoms,
                self.critic_criterion
            ) for x in atoms_t
        ]

        return policy_loss, value_loss

    def _add_noise_to_actions(self, actions):
        action_noise = torch.normal(
            mean=torch.zeros_like(actions), std=self.action_noise_std
        )
        action_noise = action_noise.clamp(
            -self.action_noise_clip, self.action_noise_clip
        )
        actions = actions + action_noise
        actions = actions.clamp(self.min_action, self.max_action)
        return actions

    def critic_update(self, loss):
        metrics = {}
        for i in range(len(self.critics)):
            self.critics[i].zero_grad()
            self.critics_optimizer[i].zero_grad()
            loss[i].backward()
            if self.critic_grad_clip_fn is not None:
                self.critic_grad_clip_fn(self.critics[i].parameters())
            self.critics_optimizer[i].step()
            if self.critics_scheduler[i] is not None:
                self.critics_scheduler[i].step()
                lr = self.critics_scheduler[i].get_lr()[0]
                metrics[f"lr_critic{i}"] = lr
        return metrics

    def target_critic_update(self):
        for target, source in zip(self.target_critics, self.critics):
            soft_update(target, source, self.critic_tau)

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

        loss = policy_loss
        for l_ in value_loss:
            loss += l_

        metrics = {
            f"loss_critic{i}": x.item()
            for i, x in enumerate(value_loss)
        }
        metrics = {
            **{
                "loss": loss.item(),
                "loss_actor": policy_loss.item()
            },
            **metrics
        }
        metrics = {**metrics, **actor_update_metrics, **critic_update_metrics}

        return metrics

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
                    value2_i = value2[i]
                    if value2_i is not None:
                        value2_i = value2_i.state_dict()
                        checkpoint[f"{key2}{i}_state_dict"] = value2_i

        return checkpoint

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

    @classmethod
    def prepare_for_trainer(cls, config):
        # hack to prevent cycle dependencies
        from catalyst.contrib.registry import Registry

        config_ = config.copy()

        actor_state_shape = (
            config_["shared"]["history_len"],
            config_["shared"]["observation_size"],
        )
        actor_action_size = config_["shared"]["action_size"]
        n_step = config_["shared"]["n_step"]
        gamma = config_["shared"]["gamma"]
        history_len = config_["shared"]["history_len"]
        trainer_state_shape = (config_["shared"]["observation_size"], )
        trainer_action_shape = (config_["shared"]["action_size"], )

        actor_fn = config_["actor"].pop("agent", None)
        actor = Registry.get_agent(
            agent=actor_fn,
            state_shape=actor_state_shape,
            action_size=actor_action_size,
            **config_["actor"]
        )

        critic_fn = config_["critic"].pop("agent", None)
        critic = Registry.get_agent(
            agent=critic_fn,
            state_shape=actor_state_shape,
            action_size=actor_action_size,
            **config_["critic"]
        )

        n_critics = config_["algorithm"].pop("n_critics", 2)
        critics = [
            Registry.get_agent(
                agent=critic_fn,
                state_shape=actor_state_shape,
                action_size=actor_action_size,
                **config_["critic"]
            ) for _ in range(n_critics - 1)
        ]

        algorithm = cls(
            **config_["algorithm"],
            actor=actor,
            critic=critic,
            critics=critics,
            n_step=n_step,
            gamma=gamma
        )

        kwargs = {
            "algorithm": algorithm,
            "state_shape": trainer_state_shape,
            "action_shape": trainer_action_shape,
            "n_step": n_step,
            "gamma": gamma,
            "history_len": history_len
        }

        return kwargs

    @classmethod
    def prepare_for_sampler(cls, config):
        # hack to prevent cycle dependencies
        from catalyst.contrib.registry import Registry

        config_ = config.copy()

        actor_state_shape = (
            config_["shared"]["history_len"],
            config_["shared"]["observation_size"],
        )
        actor_action_size = config_["shared"]["action_size"]

        actor_fn = config_["actor"].pop("agent", None)
        actor = Registry.get_agent(
            agent=actor_fn,
            state_shape=actor_state_shape,
            action_size=actor_action_size,
            **config_["actor"]
        )

        history_len = config_["shared"]["history_len"]

        kwargs = {"actor": actor, "history_len": history_len}

        return kwargs
