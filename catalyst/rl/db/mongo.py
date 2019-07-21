import datetime
import pymongo
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
        self._flag_collection = self._agent_db["flag"]
        self._last_datetime = datetime.datetime.min

        self._epoch = 0
        self._sync_epoch = sync_epoch

    @property
    def training_enabled(self) -> bool:
        flag_obj = self._flag_collection.find_one(
            {"prefix": {
                "$eq": "training_flag"
            }}
        )
        flag = flag_obj.get("training_flag")
        flag = flag if flag is not None else 1  # enabled by default
        flag = int(flag) == int(1)
        return flag

    @property
    def sampling_enabled(self) -> bool:
        flag_obj = self._flag_collection.find_one(
            {"prefix": {
                "$eq": "sampling_flag"
            }}
        )
        flag = flag_obj.get("sampling_flag")
        flag = flag if flag is not None else -1  # disabled by default
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
        if message in [
            DBSpec.Message.ENABLE_SAMPLING,
            DBSpec.Message.DISABLE_SAMPLING
        ]:
            flag = message == DBSpec.Message.ENABLE_SAMPLING
            self._flag_collection.replace_one(
                {"prefix": "sampling_flag"}, {
                    "sampling_flag": flag,
                    "prefix": "sampling_flag"
                },
                upsert=True
            )
        elif message in [
            DBSpec.Message.ENABLE_TRAINING,
            DBSpec.Message.DISABLE_TRAINING
        ]:
            flag = message == DBSpec.Message.ENABLE_TRAINING
            if not flag:
                self._flag_collection.replace_one(
                    {"prefix": "sampling_flag"}, {
                        "sampling_flag": 0,
                        "prefix": "sampling_flag"
                    },
                    upsert=True
                )
            self._flag_collection.replace_one(
                {"prefix": "training_flag"}, {
                    "training_flag": flag,
                    "prefix": "training_flag"
                },
                upsert=True
            )
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
