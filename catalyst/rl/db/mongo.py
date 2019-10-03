import time
import datetime

import pymongo
import gridfs
import safitty

from catalyst.rl import utils
from catalyst.rl.core import DBSpec


class MongoDB(DBSpec):
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 12000,
        prefix: str = None,
        sync_epoch: bool = False,
        reconnect_timeout: int = 3,
    ):
        self._server = pymongo.MongoClient(host=host, port=port)
        self._prefix = "" if prefix is None else prefix
        self._reconnect_timeout = reconnect_timeout

        self._shared_db = self._server["shared"]
        self._agent_db = self._server[f"agent_{self._prefix}"]

        self._trajectory_collection = self._shared_db["trajectories"]
        self._raw_trajectory_collection = self._shared_db["raw_trajectories"]
        self._checkpoint_collection =\
            gridfs.GridFS(self._agent_db, collection="checkpoints")
        self._message_collection = self._agent_db["messages"]

        self._last_datetime = datetime.datetime.min

        self._epoch = 0
        self._sync_epoch = sync_epoch

    def _set_flag(self, key, value):
        try:
            self._message_collection.replace_one(
                {"key": key},
                {"key": key, "value": value},
                upsert=True
            )
        except pymongo.errors.AutoReconnect:
            time.sleep(self._reconnect_timeout)
            return self._set_flag(key, value)

    def _get_flag(self, key, default=None):
        try:
            flag_obj = self._message_collection.find_one(
                {"key": {"$eq": key}}
            )
        except pymongo.errors.AutoReconnect:
            time.sleep(self._reconnect_timeout)
            return self._get_flag(key, default)
        flag = safitty.get(flag_obj, "value", default=default)
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
        num_trajectories = self._trajectory_collection.count() - 1
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
        try:
            trajectory_ = utils.structed2dict_trajectory(trajectory)
            trajectory_ = utils.pack(trajectory_)
            collection = self._raw_trajectory_collection if raw \
                else self._trajectory_collection

            collection.insert_one(
                {
                    "trajectory": trajectory_,
                    "date": datetime.datetime.utcnow(),
                    "epoch": self._epoch
                }
            )
        except pymongo.errors.AutoReconnect:
            time.sleep(self._reconnect_timeout)
            return self.put_trajectory(trajectory, raw)

    def get_trajectory(self, index=None):
        assert index is None

        try:
            trajectory_obj = self._trajectory_collection.find_one(
                {"date": {
                    "$gt": self._last_datetime
                }}
            )
        except pymongo.errors.AutoReconnect:
            time.sleep(self._reconnect_timeout)
            return self.get_trajectory(index)

        if trajectory_obj is not None:
            self._last_datetime = trajectory_obj["date"]

            trajectory, trajectory_epoch = \
                utils.unpack(trajectory_obj["trajectory"]), \
                trajectory_obj["epoch"]
            if self._sync_epoch and self._epoch != trajectory_epoch:
                trajectory = None
            else:
                trajectory = utils.dict2structed_trajectory(trajectory)
        else:
            trajectory = None

        return trajectory

    def del_trajectory(self):
        try:
            self._trajectory_collection.drop()
        except pymongo.errors.AutoReconnect:
            time.sleep(self._reconnect_timeout)
            return self.del_trajectory()

    def put_checkpoint(self, checkpoint, epoch):
        try:
            self._epoch = epoch
            checkpoint_ = utils.pack(checkpoint)
            if self._checkpoint_collection.exists({"filename": "checkpoint"}):
                self.del_checkpoint()

            self._checkpoint_collection.put(
                checkpoint_,
                encoding="ascii",
                filename="checkpoint",
                epoch=self._epoch
            )

        except pymongo.errors.AutoReconnect:
            time.sleep(self._reconnect_timeout)
            return self.put_checkpoint(checkpoint, epoch)

    def get_checkpoint(self):
        try:
            checkpoint_obj = self._checkpoint_collection.find_one(
                {"filename": "checkpoint"}
            )
        except pymongo.errors.AutoReconnect:
            time.sleep(self._reconnect_timeout)
            return self.get_checkpoint()

        if checkpoint_obj is not None:
            checkpoint = checkpoint_obj.read().decode("ascii")
            self._epoch = checkpoint_obj.epoch
            checkpoint = utils.unpack(checkpoint)
        else:
            checkpoint = None
        return checkpoint

    def del_checkpoint(self):
        id_ = self._checkpoint_collection.find_one(
            {"filename": "checkpoint"}
        )._id
        self._checkpoint_collection.delete(id_)


__all__ = ["MongoDB"]
