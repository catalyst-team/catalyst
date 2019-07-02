from abc import abstractmethod, ABC


class DBSpec(ABC):

    @property
    @abstractmethod
    def epoch(self) -> int:
        pass

    @property
    @abstractmethod
    def num_trajectories(self) -> int:
        pass

    @abstractmethod
    def set_sample_flag(self, sample: bool):
        pass

    @abstractmethod
    def get_sample_flag(self) -> bool:
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
