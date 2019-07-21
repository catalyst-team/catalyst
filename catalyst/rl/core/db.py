from abc import abstractmethod, ABC
from enum import Enum


class DBSpec(ABC):

    class Message(Enum):
        ENABLE_TRAINING = 0
        DISABLE_TRAINING = 1
        ENABLE_SAMPLING = 2
        DISABLE_SAMPLING = 3

    @property
    @abstractmethod
    def training_enabled(self) -> bool:
        pass

    @property
    @abstractmethod
    def sampling_enabled(self) -> bool:
        pass

    @property
    @abstractmethod
    def epoch(self) -> int:
        pass

    @property
    @abstractmethod
    def num_trajectories(self) -> int:
        pass

    @abstractmethod
    def push_message(self, message: Message):
        pass

    @abstractmethod
    def push_trajectory(self, trajectory, raw: bool):
        pass

    @abstractmethod
    def get_trajectory(self, index=None):
        pass

    @abstractmethod
    def clean_trajectories(self):
        pass

    @abstractmethod
    def save_checkpoint(self, checkpoint, epoch):
        pass

    @abstractmethod
    def load_checkpoint(self):
        pass

    @abstractmethod
    def clean_checkpoint(self):
        pass
