from abc import abstractmethod, ABC


class DBSpec(ABC):

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
    def push_trajectory(self, trajectory):
        pass

    @abstractmethod
    def get_trajectory(self, index=None):
        pass

    @abstractmethod
    def clean_trajectories(self):
        pass

    @abstractmethod
    def dump_weights(self, weights, prefix, epoch):
        pass

    @abstractmethod
    def load_weights(self, prefix):
        pass

    @abstractmethod
    def clean_weights(self, prefix):
        pass
