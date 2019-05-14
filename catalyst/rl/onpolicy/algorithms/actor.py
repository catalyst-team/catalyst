from typing import Union, Dict
import torch

from catalyst.dl.utils import UtilsFactory
from catalyst.rl.registry import AGENTS
from catalyst.rl.agents.core import ActorSpec, CriticSpec
from catalyst.rl.environments.core import EnvironmentSpec
from catalyst.rl.offpolicy.algorithms.utils import get_agent_stuff_from_params
from .core import AlgorithmSpec


class ActorAlgorithmSpec(AlgorithmSpec):
    def __init__(
        self,
        actor: ActorSpec,
        gamma: float,
        n_step: int,
        actor_loss_params: Dict = None,
        actor_optimizer_params: Dict = None,
        actor_scheduler_params: Dict = None,
        actor_grad_clip_params: Dict = None,
        **kwargs
    ):
        self._device = UtilsFactory.prepare_device()

        self.actor = actor.to(self._device)

        # actor preparation
        actor_stuff = get_agent_stuff_from_params(
            agent=self.actor,
            loss_params=actor_loss_params,
            optimizer_params=actor_optimizer_params,
            scheduler_params=actor_scheduler_params,
            grad_clip_params=actor_grad_clip_params
        )
        # criterion
        self._actor_loss_params = actor_stuff["loss_params"]
        self.actor_criterion = actor_stuff["criterion"]
        # optimizer
        self._actor_optimizer_params = actor_stuff["optimizer_params"]
        self.actor_optimizer = actor_stuff["optimizer"]
        # scheduler
        self._actor_scheduler_params = actor_stuff["scheduler_params"]
        self.actor_scheduler = actor_stuff["scheduler"]
        # grad clipping
        self._actor_grad_clip_params = actor_stuff["grad_clip_params"]
        self.actor_grad_clip_fn = actor_stuff["grad_clip_fn"]

        # other hyperparameters
        self._n_step = n_step
        self._gamma = gamma

        # other init
        self._init(**kwargs)

    def _to_tensor(self, *args, **kwargs):
        return torch.Tensor(*args, **kwargs).to(self._device)

    @property
    def n_step(self) -> int:
        return self._n_step

    @property
    def gamma(self) -> float:
        return self._gamma

    def pack_checkpoint(self):
        checkpoint = {}

        for key in ["actor"]:
            checkpoint[f"{key}_state_dict"] = getattr(self, key).state_dict()
            for key2 in ["optimizer", "scheduler"]:
                key2 = f"{key}_{key2}"
                value2 = getattr(self, key2, None)
                if value2 is not None:
                    checkpoint[f"{key2}_state_dict"] = value2.state_dict()

        return checkpoint

    def unpack_checkpoint(self, checkpoint):
        raise NotImplementedError()

    def save_checkpoint(self, filepath):
        raise NotImplementedError()

    def load_checkpoint(self, filepath, load_optimizer=True):
        checkpoint = UtilsFactory.load_checkpoint(filepath)
        for key in ["actor"]:
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
        pass

    def postprocess_buffer(self, buffers, len):
        pass

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

        algorithm = cls(
            **config_["algorithm"],
            actor=actor,
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
