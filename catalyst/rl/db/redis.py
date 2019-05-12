from redis import StrictRedis
from catalyst.utils.compression import pack, unpack
from .core import DBSpec


class RedisDB(DBSpec):
    def __init__(self, port=12000, prefix=None, sync_epoch=False):
        self._server = StrictRedis(port=port)
        self._prefix = "" if prefix is None else prefix

        self._index = 0
        self._epoch = 0
        self._sync_epoch = sync_epoch

    @property
    def num_trajectories(self) -> int:
        num_trajectories = self._server.llen("trajectories") - 1
        return num_trajectories

    def set_sample_flag(self, sample: bool):
        self._server.set("sample_flag", int(sample))

    def get_sample_flag(self) -> bool:
        flag = int(self._server.get("sample_flag") or -1) == int(1)
        return flag

    def push_trajectory(self, trajectory):
        trajectory = {
            "trajectory": trajectory,
            "epoch": self._epoch
        }
        trajectory = pack(trajectory)
        self._server.rpush("trajectories", trajectory)

    def get_trajectory(self, index=None):
        index = index if index is not None else self._index
        trajectory = self._server.lindex("trajectories", index)
        if trajectory is not None:
            self._index = index + 1

            trajectory = unpack(trajectory)
            trajectory, trajectory_epoch = \
                trajectory["trajectory"], trajectory["epoch"]
            if self._sync_epoch and self._epoch != trajectory_epoch:
                trajectory = None

        return trajectory

    def clean_trajectories(self):
        self._server.delete("trajectories")
        self._index = 0

    def dump_weights(self, weights, prefix, epoch):
        self._epoch = epoch
        weights = {
            "weights": weights,
            "epoch": self._epoch
        }
        weights = pack(weights)
        self._server.set(f"{self._prefix}_{prefix}_weights", weights)

    def load_weights(self, prefix):
        weights = self._server.get(f"{self._prefix}_{prefix}_weights")
        if weights is not None:
            weights = unpack(weights)
        self._epoch = weights["epoch"]
        return weights["weights"]

    def clean_weights(self, prefix):
        self._server.delete(f"{self._prefix}_{prefix}_weights")
