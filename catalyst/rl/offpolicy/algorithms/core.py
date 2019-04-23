from typing import Union, Dict
from abc import abstractmethod, ABC
from catalyst.rl.agents.core import ActorSpec, CriticSpec
from catalyst.rl.environments.core import EnvironmentSpec


class AlgorithmSpec(ABC):

    @property
    @abstractmethod
    def n_step(self) -> int:
        pass

    @property
    @abstractmethod
    def gamma(self) -> float:
        pass

    @abstractmethod
    def pack_checkpoint(self):
        pass

    @abstractmethod
    def unpack_checkpoint(self, checkpoint):
        pass

    @abstractmethod
    def save_checkpoint(self, filepath):
        pass

    @abstractmethod
    def load_checkpoint(self, filepath):
        pass

    @abstractmethod
    def actor_update(self, loss):
        pass

    @abstractmethod
    def critic_update(self, loss):
        pass

    @abstractmethod
    def target_actor_update(self):
        pass

    @abstractmethod
    def target_critic_update(self):
        pass

    @abstractmethod
    def train(self, batch, actor_update=True, critic_update=True):
        pass

    @classmethod
    @abstractmethod
    def prepare_for_trainer(cls, env_spec: EnvironmentSpec, config: Dict):
        pass

    @classmethod
    @abstractmethod
    def prepare_for_sampler(
        cls,
        env_spec: EnvironmentSpec,
        config: Dict
    ) -> Union[ActorSpec, CriticSpec]:
        pass
