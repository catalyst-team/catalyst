from redis import StrictRedis
from catalyst.utils.serialization import serialize, deserialize
from .core import DBSpec


class RedisDB(DBSpec):
    def __init__(self, port, prefix=None):
        self._server = StrictRedis(port=port)
        self._prefix = "" if prefix is None else prefix

    @property
    def num_trajectories(self) -> int:
        redis_len = self._server.llen("trajectories") - 1
        return redis_len

    def push_trajectory(self, trajectory):
        trajectory = serialize(trajectory)
        self._server.rpush("trajectories", trajectory)

    def get_trajectory(self, index):
        trajectory = deserialize(self._server.lindex("trajectories", index))
        return trajectory

    def dump_weights(self, weights, suffix):
        weights = serialize(weights)
        self._server.set(f"{self._prefix}_{suffix}_weights", weights)

    def load_weights(self, suffix):
        weights = self._server.get(f"{self._prefix}_{suffix}_weights")
        if weights is None:
            return None
        weights = deserialize(weights)
        return weights
