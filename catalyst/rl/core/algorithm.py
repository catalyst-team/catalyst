from typing import Union, Dict
from abc import abstractmethod, ABC
from .agent import ActorSpec, CriticSpec
from .environment import EnvironmentSpec


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
    def _init(self, **kwargs):
        pass

    @abstractmethod
    def pack_checkpoint(self, with_optimizer: bool = True, **kwargs):
        pass

    @abstractmethod
    def unpack_checkpoint(
        self,
        checkpoint,
        with_optimizer: bool = True,
        **kwargs
    ):
        pass

    @abstractmethod
    def train(self, batch: Dict, **kwargs):
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
