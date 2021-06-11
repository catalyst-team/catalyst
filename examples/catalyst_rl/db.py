# flake8: noqa
from abc import ABC, abstractmethod
import datetime
from enum import Enum
import time

try:
    from redis import Redis

    IS_REDIS_AVAILABLE = True
except ImportError:
    IS_REDIS_AVAILABLE = False

try:
    import gridfs
    import pymongo

    IS_MONGO_AVAILABLE = True
except ImportError:
    IS_MONGO_AVAILABLE = False

from misc import dict2structed_trajectory, structed2dict_trajectory, Trajectory

from catalyst.utils import pack, unpack


class IRLDatabaseMessage(Enum):
    ENABLE_TRAINING = 0
    DISABLE_TRAINING = 1
    ENABLE_SAMPLING = 2
    DISABLE_SAMPLING = 3


class IRLDatabase(ABC):
    @property
    @abstractmethod
    def training_enabled(self) -> bool:
        pass

    @property
    @abstractmethod
    def sampling_enabled(self) -> bool:
        pass

    @property
    @abstractmethod
    def epoch(self) -> int:
        pass

    @property
    @abstractmethod
    def num_trajectories(self) -> int:
        pass

    @abstractmethod
    def add_message(self, message: IRLDatabaseMessage):
        pass

    @abstractmethod
    def add_trajectory(self, trajectory: Trajectory):
        pass

    @abstractmethod
    def get_trajectory(self, index=None) -> Trajectory:
        pass

    @abstractmethod
    def del_trajectory(self):
        pass

    @abstractmethod
    def add_checkpoint(self, checkpoint, epoch):
        pass

    @abstractmethod
    def get_checkpoint(self):
        pass

    @abstractmethod
    def del_checkpoint(self):
        pass


if IS_REDIS_AVAILABLE:

    class RedisDB(IRLDatabase):
        def __init__(self, host="127.0.0.1", port=12000, prefix=None, sync_epoch=False):
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

        def add_message(self, message: IRLDatabaseMessage):
            if message == IRLDatabaseMessage.ENABLE_SAMPLING:
                self._set_flag("sampling_flag", 1)
            elif message == IRLDatabaseMessage.DISABLE_SAMPLING:
                self._set_flag("sampling_flag", 0)
            elif message == IRLDatabaseMessage.DISABLE_TRAINING:
                self._set_flag("sampling_flag", 0)
                self._set_flag("training_flag", 0)
            elif message == IRLDatabaseMessage.ENABLE_TRAINING:
                self._set_flag("training_flag", 1)
            else:
                raise NotImplementedError("unknown message", message)

        def add_trajectory(self, trajectory: Trajectory, raw=False):
            trajectory = structed2dict_trajectory(trajectory)
            trajectory = {"trajectory": trajectory, "epoch": self._epoch}
            trajectory = pack(trajectory)
            name = "raw_trajectories" if raw else "trajectories"
            self._server.rpush(name, trajectory)

        def get_trajectory(self, index=None) -> Trajectory:
            index = index if index is not None else self._index
            trajectory = self._server.lindex("trajectories", index)
            if trajectory is not None:
                self._index = index + 1

                trajectory = unpack(trajectory)
                trajectory, trajectory_epoch = trajectory["trajectory"], trajectory["epoch"]
                if self._sync_epoch and self._epoch != trajectory_epoch:
                    trajectory = None
                else:
                    trajectory = dict2structed_trajectory(trajectory)

            return trajectory

        def del_trajectory(self) -> None:
            self._server.delete("trajectories")
            self._index = 0

        def add_checkpoint(self, checkpoint, epoch):
            self._epoch = epoch
            checkpoint = {"checkpoint": checkpoint, "epoch": self._epoch}
            checkpoint = pack(checkpoint)
            self._server.set(f"{self._prefix}_checkpoint", checkpoint)

        def get_checkpoint(self):
            checkpoint = self._server.get(f"{self._prefix}_checkpoint")
            if checkpoint is None:
                return None
            checkpoint = unpack(checkpoint)
            self._epoch = checkpoint.get("epoch")
            return checkpoint["checkpoint"]

        def del_checkpoint(self):
            self._server.delete(f"{self._prefix}_weights")


if IS_MONGO_AVAILABLE:

    class MongoDB(IRLDatabase):
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
            self._checkpoint_collection = gridfs.GridFS(self._agent_db, collection="checkpoints")
            self._message_collection = self._agent_db["messages"]

            self._last_datetime = datetime.datetime.min

            self._epoch = 0
            self._sync_epoch = sync_epoch

        def _set_flag(self, key, value):
            try:
                self._message_collection.replace_one(
                    {"key": key}, {"key": key, "value": value}, upsert=True
                )
            except pymongo.errors.AutoReconnect:
                time.sleep(self._reconnect_timeout)
                return self._set_flag(key, value)

        def _get_flag(self, key, default=None):
            try:
                flag_obj = self._message_collection.find_one({"key": {"$eq": key}})
            except pymongo.errors.AutoReconnect:
                time.sleep(self._reconnect_timeout)
                return self._get_flag(key, default)
            flag = flag_obj.get("value", default)
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

        def add_message(self, message: IRLDatabaseMessage):
            if message == IRLDatabaseMessage.ENABLE_SAMPLING:
                self._set_flag("sampling_flag", 1)
            elif message == IRLDatabaseMessage.DISABLE_SAMPLING:
                self._set_flag("sampling_flag", 0)
            elif message == IRLDatabaseMessage.DISABLE_TRAINING:
                self._set_flag("sampling_flag", 0)
                self._set_flag("training_flag", 0)
            elif message == IRLDatabaseMessage.ENABLE_TRAINING:
                self._set_flag("training_flag", 1)
            else:
                raise NotImplementedError("unknown message", message)

        def add_trajectory(self, trajectory, raw=False):
            try:
                trajectory_ = structed2dict_trajectory(trajectory)
                trajectory_ = pack(trajectory_)
                collection = (
                    self._raw_trajectory_collection if raw else self._trajectory_collection
                )

                collection.insert_one(
                    {
                        "trajectory": trajectory_,
                        "date": datetime.datetime.utcnow(),
                        "epoch": self._epoch,
                    }
                )
            except pymongo.errors.AutoReconnect:
                time.sleep(self._reconnect_timeout)
                return self.add_trajectory(trajectory, raw)

        def get_trajectory(self, index=None):
            assert index is None

            try:
                trajectory_obj = self._trajectory_collection.find_one(
                    {"date": {"$gt": self._last_datetime}}
                )
            except pymongo.errors.AutoReconnect:
                time.sleep(self._reconnect_timeout)
                return self.get_trajectory(index)

            if trajectory_obj is not None:
                self._last_datetime = trajectory_obj["date"]

                trajectory, trajectory_epoch = (
                    unpack(trajectory_obj["trajectory"]),
                    trajectory_obj["epoch"],
                )
                if self._sync_epoch and self._epoch != trajectory_epoch:
                    trajectory = None
                else:
                    trajectory = dict2structed_trajectory(trajectory)
            else:
                trajectory = None

            return trajectory

        def del_trajectory(self):
            try:
                self._trajectory_collection.drop()
            except pymongo.errors.AutoReconnect:
                time.sleep(self._reconnect_timeout)
                return self.del_trajectory()

        def add_checkpoint(self, checkpoint, epoch):
            try:
                self._epoch = epoch
                checkpoint_ = pack(checkpoint)
                if self._checkpoint_collection.exists({"filename": "checkpoint"}):
                    self.del_checkpoint()

                self._checkpoint_collection.put(
                    checkpoint_, encoding="ascii", filename="checkpoint", epoch=self._epoch
                )

            except pymongo.errors.AutoReconnect:
                time.sleep(self._reconnect_timeout)
                return self.add_checkpoint(checkpoint, epoch)

        def get_checkpoint(self):
            try:
                checkpoint_obj = self._checkpoint_collection.find_one({"filename": "checkpoint"})
            except pymongo.errors.AutoReconnect:
                time.sleep(self._reconnect_timeout)
                return self.get_checkpoint()

            if checkpoint_obj is not None:
                checkpoint = checkpoint_obj.read().decode("ascii")
                self._epoch = checkpoint_obj.epoch
                checkpoint = unpack(checkpoint)
            else:
                checkpoint = None
            return checkpoint

        def del_checkpoint(self):
            id_ = self._checkpoint_collection.find_one({"filename": "checkpoint"})._id
            self._checkpoint_collection.delete(id_)
