import datetime

import pymongo
import safitty

from catalyst.rl import utils
from catalyst.rl.core import DBSpec


class MongoDB(DBSpec):
    def __init__(
        self, host="127.0.0.1", port=12000, prefix=None, sync_epoch=False
    ):
        self._server = pymongo.MongoClient(host=host, port=port)
        self._prefix = "" if prefix is None else prefix

        self._shared_db = self._server["shared"]
        self._agent_db = self._server[f"agent_{self._prefix}"]

        self._trajectory_collection = self._shared_db["trajectories"]
        self._raw_trajectory_collection = self._shared_db["raw_trajectories"]
        self._checkpoints_collection = self._agent_db["checkpoints"]
        self._messages_collection = self._agent_db["messages"]
        self._last_datetime = datetime.datetime.min

        self._epoch = 0
        self._sync_epoch = sync_epoch

    def _set_flag(self, key, value):
        self._messages_collection.replace_one(
            {"key": key},
            {"key": key, "value": value},
            upsert=True
        )

    def _get_flag(self, key, default=None):
        flag_obj = self._messages_collection.find_one({"key": {"$eq": key}})
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

    def push_trajectory(self, trajectory, raw=False):
        trajectory = utils.structed2dict_trajectory(trajectory)
        trajectory = utils.pack(trajectory)
        collection = self._raw_trajectory_collection if raw \
            else self._trajectory_collection

        collection.insert_one(
            {
                "trajectory": trajectory,
                "date": datetime.datetime.utcnow(),
                "epoch": self._epoch
            }
        )

    def get_trajectory(self, index=None):
        assert index is None

        trajectory_obj = self._trajectory_collection.find_one(
            {"date": {
                "$gt": self._last_datetime
            }}
        )
        if trajectory_obj is not None:
            self._last_datetime = trajectory_obj["date"]

            trajectory, trajectory_epoch = \
                utils.unpack(
                    trajectory_obj["trajectory"]), trajectory_obj["epoch"]
            if self._sync_epoch and self._epoch != trajectory_epoch:
                trajectory = None
            else:
                trajectory = utils.dict2structed_trajectory(trajectory)
        else:
            trajectory = None

        return trajectory

    def clean_trajectories(self):
        self._trajectory_collection.drop()

    def save_checkpoint(self, checkpoint, epoch):
        self._epoch = epoch

        checkpoint = utils.pack(checkpoint)
        self._checkpoints_collection.replace_one(
            {"prefix": "checkpoint"}, {
                "checkpoint": checkpoint,
                "prefix": "checkpoint",
                "epoch": self._epoch
            },
            upsert=True
        )

    def load_checkpoint(self):
        checkpoint_obj = self._checkpoints_collection.find_one(
            {"prefix": "checkpoint"}
        )
        checkpoint = checkpoint_obj.get("checkpoint")
        if checkpoint is None:
            return None
        self._epoch = checkpoint_obj["epoch"]
        checkpoint = utils.unpack(checkpoint)
        return checkpoint

    def clean_checkpoint(self):
        self._checkpoints_collection.delete_one({"prefix": "checkpoint"})


__all__ = ["MongoDB"]
