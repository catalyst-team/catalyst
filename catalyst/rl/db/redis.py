from redis import StrictRedis
from catalyst.utils.serialization import serialize, deserialize
from catalyst.utils.compression import compress, decompress, LZ4_ENABLED
from .core import DBSpec


class RedisDB(DBSpec):
    def __init__(self, port=12000, prefix=None, use_compression=True):
        self._server = StrictRedis(port=port)
        self._prefix = "" if prefix is None else prefix

        self.index = 0

        self._use_compression = use_compression and LZ4_ENABLED
        if self._use_compression:
            self._pack = compress
            self._unpack = decompress
        else:
            self._pack = serialize
            self._unpack = deserialize

    @property
    def num_trajectories(self) -> int:
        num_trajectories = self._server.llen("trajectories") - 1
        return num_trajectories

    def push_trajectory(self, trajectory):
        trajectory = self._pack(trajectory)
        self._server.rpush("trajectories", trajectory)

    def get_trajectory(self, index=None):
        index = index if index is not None else self.index
        trajectory = self._server.lindex("trajectories", index)
        if trajectory is not None:
            trajectory = self._unpack(trajectory)
            self.index = index + 1
        return trajectory

    def dump_weights(self, weights, prefix):
        weights = self._pack(weights)
        self._server.set(f"{self._prefix}_{prefix}_weights", weights)

    def load_weights(self, prefix):
        weights = self._server.get(f"{self._prefix}_{prefix}_weights")
        if weights is not None:
            weights = self._unpack(weights)
        return weights
