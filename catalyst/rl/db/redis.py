from redis import Redis
from catalyst.rl import utils
from catalyst.rl.core import DBSpec


class RedisDB(DBSpec):
    def __init__(
        self,
        host="127.0.0.1",
        port=12000,
        prefix=None,
        sync_epoch=False
    ):
        self._server = Redis(host=host, port=port)
        self._prefix = "" if prefix is None else prefix

        self._index = 0
        self._epoch = 0
        self._sync_epoch = sync_epoch

    @property
    def epoch(self) -> int:
        return self._epoch

    @property
    def num_trajectories(self) -> int:
        num_trajectories = self._server.llen("trajectories") - 1
        return num_trajectories

    def set_sample_flag(self, sample: bool):
        self._server.set("sample_flag", int(sample))

    def get_sample_flag(self) -> bool:
        flag = int(self._server.get("sample_flag") or -1) == int(1)
        return flag

    def push_trajectory(self, trajectory, raw=False):
        trajectory = utils.structed2dict_trajectory(trajectory)
        trajectory = {
            "trajectory": trajectory,
            "epoch": self._epoch
        }
        trajectory = utils.pack(trajectory)
        name = "raw_trajectories" if raw else "trajectories"
        self._server.rpush(name, trajectory)

    def get_trajectory(self, index=None):
        index = index if index is not None else self._index
        trajectory = self._server.lindex("trajectories", index)
        if trajectory is not None:
            self._index = index + 1

            trajectory = utils.unpack(trajectory)
            trajectory, trajectory_epoch = \
                trajectory["trajectory"], trajectory["epoch"]
            if self._sync_epoch and self._epoch != trajectory_epoch:
                trajectory = None
            else:
                trajectory = utils.dict2structed_trajectory(trajectory)

        return trajectory

    def clean_trajectories(self):
        self._server.delete("trajectories")
        self._index = 0

    def save_checkpoint(self, checkpoint, epoch):
        self._epoch = epoch
        checkpoint = {
            "checkpoint": checkpoint,
            "epoch": self._epoch
        }
        checkpoint = utils.pack(checkpoint)
        self._server.set(f"{self._prefix}_checkpoint", checkpoint)

    def load_checkpoint(self):
        checkpoint = self._server.get(f"{self._prefix}_checkpoint")
        if checkpoint is None:
            return None
        checkpoint = utils.unpack(checkpoint)
        self._epoch = checkpoint.get("epoch")
        return checkpoint["checkpoint"]

    def clean_checkpoint(self):
        self._server.delete(f"{self._prefix}_weights")


__all__ = ["RedisDB"]
