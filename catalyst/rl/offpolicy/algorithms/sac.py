import copy
import torch
import torch.nn.functional as F

from catalyst.rl.registry import SCHEDULERS, OPTIMIZERS, AGENTS
from catalyst.dl.utils import UtilsFactory
from catalyst.rl.offpolicy.algorithms.core import Algorithm
from catalyst.rl.offpolicy.algorithms.utils import categorical_loss, \
    quantile_loss, soft_update
from catalyst.utils.model import prepare_optimizable_params


class SAC(Algorithm):
    def _init(
        self,
        critics,
        reward_scale=1.0,
        values_range=(-10., 10.),
        critic_distribution=None,
        **kwargs
    ):
        """
        Parameters
        ----------
        reward_scale: float,
            THE MOST IMPORTANT HYPERPARAMETER which controls the ratio
            between maximizing rewards and acting as randomly as possible
        use_regularization: bool,
            whether to use l2 regularization on policy network outputs,
            regularization can not be used with RealNVPActor
        mu_and_sigma_reg: float,
            coefficient for l2 regularization on mu and log_sigma
        policy_grad_estimator: str,
            "reinforce": may be used with arbitrary explicit policy
            "reparametrization_trick": may be used with reparametrizable
            policy, e.g. Gaussian, normalizing flow (Real NVP).
        """
        super()._init(**kwargs)

        self.n_atoms = self.critic.out_features
        self._loss_fn = self._base_loss

        self.reward_scale = reward_scale
        # @TODO: policy regularization

        critics = [x.to(self._device) for x in critics]
        critics_optimizer = [
            OPTIMIZERS.get_from_params(
                **self.critic_optimizer_params,
                params=prepare_optimizable_params(x)
            )
            for x in critics
        ]
        critics_scheduler = [
            SCHEDULERS.get_from_params(
                **self.critic_scheduler_params,
                optimizer=x
            )
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
        actions_tp0, log_pi_tp0 = self.actor(states_t, with_log_pi=True)
        log_pi_tp0 = log_pi_tp0 / self.reward_scale
        q_values_tp0 = [x(states_t, actions_tp0) for x in self.critics]
        q_values_tp0_min = torch.cat(q_values_tp0, dim=-1).min(dim=-1)[0]
        policy_loss = torch.mean(log_pi_tp0 - q_values_tp0_min)

        # critic loss
        actions_tp1, log_pi_tp1 = self.actor(
            states_tp1, with_log_pi=True
        )
        log_pi_tp1 = log_pi_tp1 / self.reward_scale
        q_values_t = [x(states_t, actions_t) for x in self.critics]
        q_values_tp1 = torch.cat(
            [x(states_tp1, actions_tp1) for x in self.target_critics], dim=-1
        )
        q_values_tp1 = q_values_tp1.min(dim=-1)[0]
        v_target_tp1 = (q_values_tp1 - log_pi_tp1).detach()
        gamma = self.gamma ** self.n_step
        q_target_t = rewards_t + (1 - done_t) * gamma * v_target_tp1[:, None]
        value_loss = [
            self.critic_criterion(x, q_target_t).mean() for x in q_values_t
        ]

        return policy_loss, value_loss

    def _categorical_loss(
        self, states_t, actions_t, rewards_t, states_tp1, done_t
    ):

        # actor loss
        actions_tp0, log_pi_tp0 = self.actor(states_t, with_log_pi=True)
        log_pi_tp0 = log_pi_tp0 / self.reward_scale
        logits_tp0 = [x(states_t, actions_tp0) for x in self.critics]
        probs_tp0 = [F.softmax(x, dim=-1) for x in logits_tp0]
        q_values_tp0 = [
            torch.sum(x * self.z, dim=-1).unsqueeze_(-1) for x in probs_tp0
        ]
        q_values_tp0_min = torch.cat(q_values_tp0, dim=-1).min(dim=-1)[0]
        policy_loss = torch.mean(log_pi_tp0 - q_values_tp0_min)

        # critic loss (kl-divergence between categorical distributions)
        actions_tp1, log_pi_tp1 = self.actor(
            states_tp1, with_log_pi=True
        )
        log_pi_tp1 = log_pi_tp1 / self.reward_scale
        logits_t = [x(states_t, actions_t) for x in self.critics]
        logits_tp1 = [x(states_tp1, actions_tp1) for x in self.target_critics]
        probs_tp1 = [F.softmax(x, dim=-1) for x in logits_tp1]
        q_values_tp1 = [
            torch.sum(x * self.z, dim=-1).unsqueeze_(-1) for x in probs_tp1
        ]
        probs_ids_tp1_min = torch.cat(q_values_tp1, dim=-1).argmin(dim=1)

        logits_tp1 = torch.cat([x.unsqueeze(-1) for x in logits_tp1], dim=-1)
        logits_tp1 = logits_tp1[range(len(logits_tp1)), :, probs_ids_tp1_min]
        gamma = self.gamma ** self.n_step
        z_target_tp1 = (self.z[None, :] - log_pi_tp1[:, None]).detach()
        atoms_target_t = rewards_t + (1 - done_t) * gamma * z_target_tp1
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
        actions_tp0, log_pi_tp0 = self.actor(states_t, with_log_pi=True)
        log_pi_tp0 = log_pi_tp0[:, None] / self.reward_scale
        atoms_tp0 = [
            x(states_t, actions_tp0).unsqueeze_(-1) for x in self.critics
        ]
        q_values_tp0_min = torch.cat(
            atoms_tp0, dim=-1
        ).mean(dim=1).min(dim=1)[0]
        policy_loss = torch.mean(log_pi_tp0 - q_values_tp0_min)

        # critic loss (quantile regression)
        actions_tp1, log_pi_tp1 = self.actor(
            states_tp1, with_log_pi=True
        )
        log_pi_tp1 = log_pi_tp1[:, None] / self.reward_scale
        atoms_t = [x(states_t, actions_t) for x in self.critics]
        atoms_tp1 = torch.cat(
            [
                x(states_tp1, actions_tp1).unsqueeze_(-1)
                for x in self.target_critics
            ],
            dim=-1
        )
        atoms_ids_tp1_min = atoms_tp1.mean(dim=1).argmin(dim=1)
        atoms_tp1 = atoms_tp1[range(len(atoms_tp1)), :, atoms_ids_tp1_min]
        gamma = self.gamma ** self.n_step
        atoms_tp1 = (atoms_tp1 - log_pi_tp1).detach()
        atoms_target_t = rewards_t + (1 - done_t) * gamma * atoms_tp1
        value_loss = [
            quantile_loss(
                x, atoms_target_t, self.tau, self.n_atoms,
                self.critic_criterion
            ) for x in atoms_t
        ]

        return policy_loss, value_loss

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

    def target_actor_update(self):
        pass

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
        config_ = config.copy()

        actor_state_shape = (
            config_["shared"]["history_len"],
            config_["shared"]["observation_size"],
        )
        actor_action_size = config_["shared"]["action_size"]
        n_step = config_["shared"]["n_step"]
        gamma = config_["shared"]["gamma"]
        history_len = config_["shared"]["history_len"]
        trainer_state_shape = (config_["shared"]["observation_size"],)
        trainer_action_shape = (config_["shared"]["action_size"],)

        actor_params = config_["actor"]
        actor = AGENTS.get_from_params(
            **actor_params,
            state_shape=actor_state_shape,
            action_size=actor_action_size
        )

        critic_params = config_["critic"]
        critic = AGENTS.get_from_params(
            **critic_params,
            state_shape=actor_state_shape,
            action_size=actor_action_size
        )

        n_critics = config_["algorithm"].pop("n_critics", 2)
        critics = [
            AGENTS.get_from_params(
                **critic_params,
                state_shape=actor_state_shape,
                action_size=actor_action_size
            ) for _ in
            range(n_critics - 1)
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
        config_ = config.copy()

        actor_state_shape = (
            config_["shared"]["history_len"],
            config_["shared"]["observation_size"],
        )
        actor_action_size = config_["shared"]["action_size"]

        actor_params = config_["actor"]
        actor = AGENTS.get_from_params(
            **actor_params,
            state_shape=actor_state_shape,
            action_size=actor_action_size
        )

        history_len = config_["shared"]["history_len"]

        kwargs = {"actor": actor, "history_len": history_len}

        return kwargs
