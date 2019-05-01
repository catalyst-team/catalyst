from redis import StrictRedis
from catalyst.utils.serialization import serialize, deserialize
from catalyst.utils.compression import compress, decompress, LZ4_ENABLED
from .core import DBSpec


class RedisDB(DBSpec):
    def __init__(self, port, prefix=None, use_compression=True):
        self._server = StrictRedis(port=port)
        self._prefix = "" if prefix is None else prefix
        self._use_compression = use_compression and LZ4_ENABLED
        if self._use_compression:
            self._pack = compress
            self._unpack = decompress
        else:
            self._pack = serialize
            self._unpack = deserialize

    @property
    def num_trajectories(self) -> int:
        redis_len = self._server.llen("trajectories") - 1
        return redis_len

    def push_trajectory(self, trajectory):
        trajectory = self._pack(trajectory)
        self._server.rpush("trajectories", trajectory)

    def get_trajectory(self, index):
        trajectory = self._unpack(self._server.lindex("trajectories", index))
        return trajectory

    def dump_weights(self, weights, suffix):
        weights = self._pack(weights)
        self._server.set(f"{self._prefix}_{suffix}_weights", weights)

    def load_weights(self, suffix):
        weights = self._server.get(f"{self._prefix}_{suffix}_weights")
        if weights is None:
            return None
        weights = self._unpack(weights)
        return weights
