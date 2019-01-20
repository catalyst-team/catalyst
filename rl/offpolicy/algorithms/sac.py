import copy
import torch
from catalyst.dl.utils import UtilsFactory
from catalyst.rl.offpolicy.algorithms.core import Algorithm


class SAC(Algorithm):
    def _init(
        self,
        critic_q1,
        critic_q2,
        reward_scale=1.0,
        policy_grad_estimator="reparametrization_trick",
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
        # hack to prevent cycle dependencies
        from catalyst.contrib.registry import Registry

        self._loss_fn = self._base_loss

        self.critic_q1 = critic_q1.to(self._device)
        self.critic_q2 = critic_q2.to(self._device)

        self.critic_q1_optimizer = Registry.get_optimizer(
            self.critic_q1, **self.critic_optimizer_params
        )
        self.critic_q2_optimizer = Registry.get_optimizer(
            self.critic_q2, **self.critic_optimizer_params
        )

        self.critic_q1_scheduler = Registry.get_scheduler(
            self.critic_q1_optimizer, **self.critic_scheduler_params
        )
        self.critic_q2_scheduler = Registry.get_scheduler(
            self.critic_q2_optimizer, **self.critic_scheduler_params
        )

        self.reward_scale = reward_scale
        # @TODO: policy regularization
        self.policy_grad_estimator = policy_grad_estimator

    def _base_loss(self, states_t, actions_t, rewards_t, states_tp1, done_t):

        # critic v loss
        actions_tp0, log_pi = self.actor(states_t, with_log_pi=True)
        log_pi = log_pi[:, None] / self.reward_scale
        values_t = self.critic(states_t)
        q1_values_tp0 = self.critic_q1(states_t, actions_tp0)
        q2_values_tp0 = self.critic_q2(states_t, actions_tp0)
        q_values_min_tp0 = torch.min(q1_values_tp0, q2_values_tp0)

        v_target_t = (q_values_min_tp0 - log_pi).detach()
        v_value_loss = self.critic_criterion(values_t, v_target_t).mean()

        # actor loss
        if self.policy_grad_estimator == "reparametrization_trick":
            policy_loss = torch.mean(log_pi - q1_values_tp0)
        elif self.policy_grad_estimator == "reinforce":
            policy_target = (log_pi - q1_values_tp0 + values_t).detach()
            policy_loss = torch.mean(
                self.reward_scale * log_pi * policy_target
            )
        else:
            raise NotImplementedError

        # critics q loss
        q1_values_t = self.critic_q1(states_t, actions_t)
        q2_values_t = self.critic_q2(states_t, actions_t)
        values_tp1 = self.target_critic(states_tp1).detach()
        gamma = self.gamma**self.n_step
        q_target_t = rewards_t + (1 - done_t) * gamma * values_tp1
        q1_value_loss = self.critic_criterion(q1_values_t, q_target_t).mean()
        q2_value_loss = self.critic_criterion(q2_values_t, q_target_t).mean()

        value_loss = [v_value_loss, q1_value_loss, q2_value_loss]

        return policy_loss, value_loss

    def update_step(
        self, policy_loss, value_loss, actor_update=True, critic_update=True
    ):
        # actor update
        actor_update_metrics = {}
        if actor_update:
            actor_update_metrics = self.actor_update(policy_loss) or {}

        # critic update
        v_value_loss, q1_value_loss, q2_value_loss = value_loss
        critic_update_metrics = {}
        if critic_update:
            critic_update_metrics = {
                **self.critic_v_update(v_value_loss),
                **self.critic_q1_update(q1_value_loss),
                **self.critic_q2_update(q2_value_loss)
            } or {}

        loss = v_value_loss + q1_value_loss + q2_value_loss + policy_loss
        metrics = {
            "loss": loss.item(),
            "loss_critic_v": v_value_loss.item(),
            "loss_critic_q1": q1_value_loss.item(),
            "loss_critic_q2": q2_value_loss.item(),
            "loss_actor": policy_loss.item()
        }
        metrics = {**metrics, **actor_update_metrics, **critic_update_metrics}

        return metrics

    def target_actor_update(self):
        pass

    def critic_v_update(self, loss):
        self.critic.zero_grad()
        self.critic_optimizer.zero_grad()
        loss.backward()
        if self.critic_grad_clip_fn is not None:
            self.critic_grad_clip_fn(self.critic.parameters())
        self.critic_optimizer.step()
        if self.critic_scheduler is not None:
            self.critic_scheduler.step()
            return {"lr_critic_v": self.critic_scheduler.get_lr()[0]}

    def critic_q1_update(self, loss):
        self.critic_q1.zero_grad()
        self.critic_q1_optimizer.zero_grad()
        loss.backward()
        if self.critic_grad_clip_fn is not None:
            self.critic_grad_clip_fn(self.critic_q1.parameters())
        self.critic_q1_optimizer.step()
        if self.critic_q1_scheduler is not None:
            self.critic_q1_scheduler.step()
            return {"lr_critic_q2": self.critic_q1_scheduler.get_lr()[0]}

    def critic_q2_update(self, loss):
        self.critic_q2.zero_grad()
        self.critic_q2_optimizer.zero_grad()
        loss.backward()
        if self.critic_grad_clip_fn is not None:
            self.critic_grad_clip_fn(self.critic_q2.parameters())
        self.critic_q2_optimizer.step()
        if self.critic_q2_scheduler is not None:
            self.critic_q2_scheduler.step()
            return {"lr_critic_q1": self.critic_q2_scheduler.get_lr()[0]}

    def prepare_checkpoint(self):
        checkpoint = super().prepare_checkpoint()

        for key in ["critic_q1", "critic_q2"]:
            checkpoint[f"{key}_state_dict"] = getattr(self, key).state_dict()
            for key2 in ["optimizer", "scheduler"]:
                key2 = f"{key}_{key2}"
                value2 = getattr(self, key2, None)
                if value2 is not None:
                    checkpoint[f"{key2}_state_dict"] = value2.state_dict()

        return checkpoint

    def load_checkpoint(self, filepath, load_optimizer=True):
        super().load_checkpoint(filepath, load_optimizer)

        checkpoint = UtilsFactory.load_checkpoint(filepath)
        for key in ["critic_q1", "critic_q2"]:
            value_l = getattr(self, key, None)
            if value_l is not None:
                value_r = checkpoint[f"{key}_state_dict"]
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
            config_["shared"]["state_size"],
        )
        actor_action_size = config_["shared"]["action_size"]
        n_step = config_["shared"]["n_step"]
        gamma = config_["shared"]["gamma"]
        history_len = config_["shared"]["history_len"]
        trainer_state_shape = (config_["shared"]["state_size"], )
        trainer_action_shape = (config_["shared"]["action_size"], )

        actor_fn = config_["actor"].pop("agent", None)
        actor = Registry.get_agent(
            agent=actor_fn,
            state_shape=actor_state_shape,
            action_size=actor_action_size,
            **config_["actor"]
        )

        critic_fn = config_["critic"].pop("agent", None)
        critic_q1 = Registry.get_agent(
            agent=critic_fn,
            state_shape=actor_state_shape,
            action_size=actor_action_size,
            **config_["critic"]
        )
        critic_q2 = Registry.get_agent(
            agent=critic_fn,
            state_shape=actor_state_shape,
            action_size=actor_action_size,
            **config_["critic"]
        )

        critic_v_fn = config_["critic_v"].pop("agent", None)
        critic_v = Registry.get_agent(
            agent=critic_v_fn,
            state_shape=actor_state_shape,
            **config_["critic_v"]
        )

        algorithm = cls(
            **config_["algorithm"],
            actor=actor,
            critic=critic_v,
            critic_q1=critic_q1,
            critic_q2=critic_q2,
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
            config_["shared"]["state_size"],
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
