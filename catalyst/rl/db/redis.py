from redis import Redis
from catalyst.rl import utils
from catalyst.rl.core import DBSpec


class RedisDB(DBSpec):
    def __init__(
        self, host="127.0.0.1", port=12000, prefix=None, sync_epoch=False
    ):
        self._server = Redis(host=host, port=port)
        self._prefix = "" if prefix is None else prefix

        self._index = 0
        self._epoch = 0
        self._sync_epoch = sync_epoch

    def _set_flag(self, key, value):
        self._server.set(f"{self._prefix}_{key}", value)

    def _get_flag(self, key, default=None):
        flag = self._server.get(f"{self._prefix}_{key}")
        flag = flag if flag is not None else default
        return flag

    @property
    def training_enabled(self) -> bool:
        flag = self._get_flag("training_flag", 1)  # enabled by default
        flag = int(flag) == int(1)
        return flag

    @property
    def sampling_enabled(self) -> bool:
        flag = self._get_flag("sampling_flag", -1)  # disabled by default
        flag = int(flag) == int(1)
        return flag

    @property
    def epoch(self) -> int:
        return self._epoch

    @property
    def num_trajectories(self) -> int:
        num_trajectories = self._server.llen("trajectories") - 1
        return num_trajectories

    def push_message(self, message: DBSpec.Message):
        if message == DBSpec.Message.ENABLE_SAMPLING:
            self._set_flag("sampling_flag", 1)
        elif message == DBSpec.Message.DISABLE_SAMPLING:
            self._set_flag("sampling_flag", 0)
        elif message == DBSpec.Message.DISABLE_TRAINING:
            self._set_flag("sampling_flag", 0)
            self._set_flag("training_flag", 0)
        elif message == DBSpec.Message.ENABLE_TRAINING:
            self._set_flag("training_flag", 1)
        else:
            raise NotImplementedError("unknown message", message)

    def put_trajectory(self, trajectory, raw=False):
        trajectory = utils.structed2dict_trajectory(trajectory)
        trajectory = {"trajectory": trajectory, "epoch": self._epoch}
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

    def del_trajectory(self):
        self._server.delete("trajectories")
        self._index = 0

    def put_checkpoint(self, checkpoint, epoch):
        self._epoch = epoch
        checkpoint = {"checkpoint": checkpoint, "epoch": self._epoch}
        checkpoint = utils.pack(checkpoint)
        self._server.set(f"{self._prefix}_checkpoint", checkpoint)

    def get_checkpoint(self):
        checkpoint = self._server.get(f"{self._prefix}_checkpoint")
        if checkpoint is None:
            return None
        checkpoint = utils.unpack(checkpoint)
        self._epoch = checkpoint.get("epoch")
        return checkpoint["checkpoint"]

    def del_checkpoint(self):
        self._server.delete(f"{self._prefix}_weights")


__all__ = ["RedisDB"]
