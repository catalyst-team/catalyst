from abc import abstractmethod, ABC


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
    def dump_checkpoint(self, filepath):
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
    def prepare_for_trainer(cls, config):
        pass

    @classmethod
    @abstractmethod
    def prepare_for_sampler(cls, config):
        pass
