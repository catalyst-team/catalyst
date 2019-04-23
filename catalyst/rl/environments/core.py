from abc import abstractmethod, ABC
from gym.spaces import Space, Discrete


class EnvironmentSpec(ABC):

    @property
    def discrete_actions(self) -> int:
        return isinstance(self.action_space, Discrete)

    @property
    def history_len(self) -> int:
        return 1

    @property
    @abstractmethod
    def observation_space(self) -> Space:
        pass

    @property
    @abstractmethod
    def state_space(self) -> Space:
        pass

    @property
    @abstractmethod
    def action_space(self) -> Space:
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def step(self, action):
        pass
