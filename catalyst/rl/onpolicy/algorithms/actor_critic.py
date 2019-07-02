from typing import Union, Dict

from catalyst.rl import utils
from catalyst.rl.registry import AGENTS
from catalyst.rl.core import AlgorithmSpec, \
    ActorSpec, CriticSpec, EnvironmentSpec


class OnpolicyActorCritic(AlgorithmSpec):
    def __init__(
        self,
        actor: ActorSpec,
        critic: CriticSpec,
        gamma: float,
        n_step: int,
        actor_loss_params: Dict = None,
        critic_loss_params: Dict = None,
        actor_optimizer_params: Dict = None,
        critic_optimizer_params: Dict = None,
        actor_scheduler_params: Dict = None,
        critic_scheduler_params: Dict = None,
        actor_grad_clip_params: Dict = None,
        critic_grad_clip_params: Dict = None,
        **kwargs
    ):
        self._device = utils.get_device()

        self.actor = actor.to(self._device)
        self.critic = critic.to(self._device)

        # actor preparation
        actor_components = utils.get_trainer_components(
            agent=self.actor,
            loss_params=actor_loss_params,
            optimizer_params=actor_optimizer_params,
            scheduler_params=actor_scheduler_params,
            grad_clip_params=actor_grad_clip_params
        )
        # criterion
        self._actor_loss_params = actor_components["loss_params"]
        self.actor_criterion = actor_components["criterion"]
        # optimizer
        self._actor_optimizer_params = actor_components["optimizer_params"]
        self.actor_optimizer = actor_components["optimizer"]
        # scheduler
        self._actor_scheduler_params = actor_components["scheduler_params"]
        self.actor_scheduler = actor_components["scheduler"]
        # grad clipping
        self._actor_grad_clip_params = actor_components["grad_clip_params"]
        self.actor_grad_clip_fn = actor_components["grad_clip_fn"]

        # critic preparation
        critic_components = utils.get_trainer_components(
            agent=self.critic,
            loss_params=critic_loss_params,
            optimizer_params=critic_optimizer_params,
            scheduler_params=critic_scheduler_params,
            grad_clip_params=critic_grad_clip_params
        )
        # criterion
        self._critic_loss_params = critic_components["loss_params"]
        self.critic_criterion = critic_components["criterion"]
        # optimizer
        self._critic_optimizer_params = critic_components["optimizer_params"]
        self.critic_optimizer = critic_components["optimizer"]
        # scheduler
        self._critic_scheduler_params = critic_components["scheduler_params"]
        self.critic_scheduler = critic_components["scheduler"]
        # grad clipping
        self._critic_grad_clip_params = critic_components["grad_clip_params"]
        self.critic_grad_clip_fn = critic_components["grad_clip_fn"]

        # other hyperparameters
        assert n_step == 1, "For now, on-policy setup works only with n-step=1"
        self._n_step = n_step
        self._gamma = gamma

        # other init
        self._init(**kwargs)

    @property
    def n_step(self) -> int:
        return self._n_step

    @property
    def gamma(self) -> float:
        return self._gamma

    def pack_checkpoint(self, with_optimizer: bool = True):
        checkpoint = {}

        for key in ["actor", "critic"]:
            checkpoint[f"{key}_state_dict"] = getattr(self, key).state_dict()
            if with_optimizer:
                for key2 in ["optimizer", "scheduler"]:
                    key2 = f"{key}_{key2}"
                    value2 = getattr(self, key2, None)
                    if value2 is not None:
                        checkpoint[f"{key2}_state_dict"] = value2.state_dict()

        return checkpoint

    def unpack_checkpoint(self, checkpoint, with_optimizer: bool = True):
        for key in ["actor", "critic"]:
            value_l = getattr(self, key, None)
            if value_l is not None:
                value_r = checkpoint[f"{key}_state_dict"]
                value_l.load_state_dict(value_r)

            if with_optimizer:
                for key2 in ["optimizer", "scheduler"]:
                    key2 = f"{key}_{key2}"
                    value_l = getattr(self, key2, None)
                    if value_l is not None:
                        value_r = checkpoint[f"{key2}_state_dict"]
                        value_l.load_state_dict(value_r)

    def actor_update(self, loss):
        self.actor.zero_grad()
        self.actor_optimizer.zero_grad()
        loss.backward()
        if self.actor_grad_clip_fn is not None:
            self.actor_grad_clip_fn(self.actor.parameters())
        self.actor_optimizer.step()
        if self.actor_scheduler is not None:
            self.actor_scheduler.step()
            return {"lr_actor": self.actor_scheduler.get_lr()[0]}

    def critic_update(self, loss):
        self.critic.zero_grad()
        self.critic_optimizer.zero_grad()
        loss.backward()
        if self.critic_grad_clip_fn is not None:
            self.critic_grad_clip_fn(self.critic.parameters())
        self.critic_optimizer.step()
        if self.critic_scheduler is not None:
            self.critic_scheduler.step()
            return {"lr_critic": self.critic_scheduler.get_lr()[0]}

    def get_rollout_spec(self) -> Dict:
        raise NotImplementedError()

    def get_rollout(self, states, actions, rewards, dones):
        raise NotImplementedError()

    def postprocess_buffer(self, buffers, len):
        raise NotImplementedError()

    @classmethod
    def prepare_for_trainer(
        cls,
        env_spec: EnvironmentSpec,
        config: Dict
    ) -> "AlgorithmSpec":
        config_ = config.copy()
        agents_config = config_["agents"]

        actor_params = agents_config["actor"]
        actor = AGENTS.get_from_params(
            **actor_params,
            env_spec=env_spec,
        )

        critic_params = agents_config["critic"]
        critic = AGENTS.get_from_params(
            **critic_params,
            env_spec=env_spec,
        )

        algorithm = cls(
            **config_["algorithm"],
            actor=actor,
            critic=critic,
        )

        return algorithm

    @classmethod
    def prepare_for_sampler(
        cls,
        env_spec: EnvironmentSpec,
        config: Dict
    ) -> Union[ActorSpec, CriticSpec]:
        config_ = config.copy()
        agents_config = config_["agents"]
        actor_params = agents_config["actor"]
        actor = AGENTS.get_from_params(
            **actor_params,
            env_spec=env_spec,
        )

        return actor
