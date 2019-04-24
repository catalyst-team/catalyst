from abc import abstractmethod, ABC
import torch.nn as nn


class ActorSpec(ABC, nn.Module):

    @property
    @abstractmethod
    def policy_type(self) -> str:
        pass

    @abstractmethod
    def forward(self, state, with_log_pi=False, deterministic=False):
        pass


class CriticSpec(ABC, nn.Module):

    @property
    @abstractmethod
    def num_outputs(self) -> int:
        pass

    @property
    @abstractmethod
    def num_atoms(self) -> int:
        pass

    @property
    @abstractmethod
    def distribution(self) -> str:
        pass

    @property
    @abstractmethod
    def values_range(self) -> tuple:
        pass

    @abstractmethod
    def forward(self, *args, **kwargs):
        pass
